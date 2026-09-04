"""
CrunchVision - PyTorch Dataset for biscuit cross-section images.

Loads images from data_vision/<class_name>/*.jpg (or .png/.jpeg). Class
labels are inferred from folder names, sorted alphabetically - use the
SAME class names as the acoustic dataset (data/<class_name>/) so acoustic
and vision predictions are directly comparable in fusion.py.

Usage:
    from vision_dataset import CrossSectionDataset, get_transforms
    train_tf, val_tf = get_transforms()
    ds = CrossSectionDataset("data_vision", transform=train_tf)
"""
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
IMAGE_SIZE = 224  # standard input size for MobileNetV2 / ResNet18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CrossSectionDataset(Dataset):
    def __init__(self, root_dir, transform=None, classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes or self._discover_classes()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = self._collect_samples()

    def _discover_classes(self):
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"{self.root_dir} not found. Collect cross-section photos first "
                f"(see capture_pair.py)."
            )
        return sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

    def _collect_samples(self):
        samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))
        if not samples:
            raise ValueError(f"No images found under {self.root_dir}/<class>/")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(image_size=IMAGE_SIZE):
    """Train transform: augmentation sized for a SMALL dataset (40-60
    images/class) - mild geometric + color jitter, nothing aggressive
    enough to destroy the aeration/crumb pattern the model needs to see.
    Val transform: deterministic resize, no augmentation.
    """
    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf
