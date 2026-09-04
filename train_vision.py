"""
CrunchVision - Transfer-learning training script.

Loads a pretrained MobileNetV2 (or ResNet18), FREEZES the convolutional
backbone, and trains only a new classification head on the cross-section
images in data_vision/<class_name>/*.jpg.

Why this approach for a small dataset (40-60 images/class), CPU-only:
- A frozen backbone means only the head's weights are trainable (a few
  thousand params, not millions) -> fast on a laptop CPU, and far less
  prone to overfitting a small dataset than fine-tuning the whole network.
- The pretrained backbone already knows general visual features (edges,
  textures, color gradients) from ImageNet; the crumb/aeration pattern
  in a cross-section photo is exactly the kind of texture those features
  transfer well to, even though ImageNet never saw a biscuit.
- Data augmentation (see vision_dataset.get_transforms) manufactures more
  effective training variety out of the limited real photos.

Reporting is deliberately conservative: every run prints val accuracy
WITH an explicit small-sample caveat, never a bare unqualified number.

Usage:
    python train_vision.py --epochs 15 --backbone mobilenet_v2
    python train_vision.py --epochs 15 --backbone resnet18
"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.model_selection import train_test_split

from vision_dataset import CrossSectionDataset, get_transforms

MODEL_FILE = "vision_model.pt"
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(backbone_name, n_classes):
    if backbone_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
        for p in model.features.parameters():
            p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)
    elif backbone_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return model


def trainable_head_params(model, backbone_name):
    return model.classifier.parameters() if backbone_name == "mobilenet_v2" else model.fc.parameters()


def stratified_split(dataset, val_frac=0.2, seed=SEED):
    labels = [label for _, label in dataset.samples]
    idx = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_frac, stratify=labels, random_state=seed
    )
    return train_idx, val_idx


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_vision")
    parser.add_argument("--backbone", default="mobilenet_v2", choices=["mobilenet_v2", "resnet18"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, val_tf = get_transforms()

    base_ds = CrossSectionDataset(args.data_dir, transform=None)
    classes = base_ds.classes
    n_classes = len(classes)
    print(f"Classes: {classes}  |  total images: {len(base_ds)}  |  device: {device}")

    for cls in classes:
        count = sum(1 for _, label in base_ds.samples if label == base_ds.class_to_idx[cls])
        if count < 15:
            print(f"  WARNING: '{cls}' has only {count} images - expect noisy val accuracy.")

    train_idx, val_idx = stratified_split(base_ds, val_frac=args.val_frac)

    # Separate dataset objects so train gets augmentation and val stays deterministic,
    # while both index into the exact same underlying (path, label) sample list.
    train_ds = CrossSectionDataset(args.data_dir, transform=train_tf, classes=classes)
    val_ds = CrossSectionDataset(args.data_dir, transform=val_tf, classes=classes)

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.backbone, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_head_params(model, args.backbone), lr=args.lr)

    print(f"\nTraining classification head only (backbone frozen)...")
    best_val_acc = 0.0
    val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(len(train_subset), 1)
        val_acc = evaluate(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)
        print(f"  epoch {epoch:2d}/{args.epochs}  train_loss={train_loss:.3f}  val_acc={val_acc:.2f}")

    print(f"\nFinal val accuracy: {val_acc:.2f}  (best during training: {best_val_acc:.2f})")
    print(f"Validated on {len(val_subset)} images across {n_classes} classes.")
    print("NOTE: this is a small-sample estimate (tens of images per class).")
    print("      Treat as an early directional signal, not a validated production metric.")
    if len(val_subset) < 20:
        print("      With this few validation images, a single misclassified photo swings")
        print("      the reported accuracy by several points - report a range, not a point estimate.")

    torch.save({
        "model_state": model.state_dict(),
        "backbone": args.backbone,
        "classes": classes,
    }, MODEL_FILE)
    print(f"\nSaved -> {MODEL_FILE}")


if __name__ == "__main__":
    main()
