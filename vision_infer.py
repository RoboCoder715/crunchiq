"""
CrunchVision - Inference on a single cross-section photo.

Loads the trained head (vision_model.pt) and predicts class + confidence
for one image. Used standalone, or imported by fusion.py / live_demo.py.

Usage:
    python vision_infer.py --image path/to/photo.jpg
"""
import argparse
import torch
import torch.nn.functional as F
from PIL import Image

from vision_dataset import get_transforms
from train_vision import build_model

MODEL_FILE = "vision_model.pt"


def load_vision_model(model_file=MODEL_FILE, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_file, map_location=device)
    model = build_model(checkpoint["backbone"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint["classes"], device


def predict_image(image_path, model=None, classes=None, device=None, model_file=MODEL_FILE):
    """Returns (predicted_class: str, confidence: float, all_probs: dict)."""
    if model is None:
        model, classes, device = load_vision_model(model_file)

    _, val_tf = get_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = val_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx].item())
    all_probs = {classes[i]: float(probs[i].item()) for i in range(len(classes))}
    return pred_class, confidence, all_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_file", default=MODEL_FILE)
    args = parser.parse_args()

    model, classes, device = load_vision_model(args.model_file)
    pred_class, confidence, all_probs = predict_image(args.image, model, classes, device)

    print(f"Predicted: {pred_class}  (confidence {confidence:.0%})")
    print("All class probabilities:")
    for c, p in sorted(all_probs.items(), key=lambda x: -x[1]):
        print(f"  {c:<20} {p:.0%}")


if __name__ == "__main__":
    main()
