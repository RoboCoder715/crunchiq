"""
CrunchVision - Synthetic cross-section image generator (FOR PIPELINE TESTING ONLY).

*** THESE ARE FAKE BISCUIT IMAGES. NOT REAL PHOTOS. ***
Use them to verify that data_vision/ -> train_vision.py -> vision_infer.py ->
fusion.py all wire together before you have real cross-section photos.
Replace with real camera captures (capture_pair.py) ASAP.

Visual model per class (simplified, programmatic):
  fresh     : uniform beige/tan background, many small round air pockets
               (bright circles), consistent crumb color — looks aerated + light
  stale     : denser, darker tan background, fewer + smaller air pockets,
               slight brown moisture gradient — looks compacted + damp
  overbaked : dark brown background, irregular large voids with dark edges,
               high contrast patches — looks over-caramelized + uneven
  broken    : mixed zones of two different crumb textures side-by-side,
               visible crack line — structurally heterogeneous cross-section

Usage:
    python generate_synthetic_vision.py               # 15/class, 3 classes
    python generate_synthetic_vision.py --count 20
    python generate_synthetic_vision.py --classes fresh stale overbaked
"""
import argparse
import os
import random
import math

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    print("PIL not installed. Run: pip install pillow --break-system-packages")
    raise

IMAGE_DIR = "data_vision"
IMAGE_SIZE = 256   # saved slightly larger than 224 so val-resize has room
SEED = 42


def _rng_seed(base, index):
    random.seed(base + index * 7919)


# ── colour palettes per class ─────────────────────────────────────────────────

def _fresh_bg():
    """Warm, pale beige — well-baked but not overdone crumb."""
    return (
        random.randint(195, 215),  # R
        random.randint(168, 188),  # G
        random.randint(120, 145),  # B
    )


def _stale_bg():
    """Slightly darker, more greyish tan — absorbed moisture, compacted."""
    return (
        random.randint(155, 178),
        random.randint(132, 155),
        random.randint(95, 118),
    )


def _overbaked_bg():
    """Deep brown — over-caramelized, Maillard reaction pushed too far."""
    return (
        random.randint(90, 125),
        random.randint(55, 80),
        random.randint(25, 50),
    )


# ── air pocket generators ─────────────────────────────────────────────────────

def _draw_fresh_pockets(draw, size, n_pockets):
    """Many small, bright round air pockets — uniform aeration."""
    for _ in range(n_pockets):
        x = random.randint(10, size - 10)
        y = random.randint(10, size - 10)
        r = random.randint(4, 12)
        brightness = random.randint(230, 255)
        color = (brightness, brightness - random.randint(0, 20), brightness - random.randint(20, 50))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color,
                     outline=(color[0] - 30, color[1] - 25, color[2] - 20), width=1)


def _draw_stale_pockets(draw, size, n_pockets):
    """Fewer, smaller, slightly darker pockets — moisture filled some voids."""
    for _ in range(n_pockets):
        x = random.randint(10, size - 10)
        y = random.randint(10, size - 10)
        r = random.randint(2, 7)
        brightness = random.randint(185, 215)
        color = (brightness, brightness - random.randint(10, 30), brightness - random.randint(30, 60))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color,
                     outline=(color[0] - 25, color[1] - 20, color[2] - 15), width=1)


def _draw_overbaked_pockets(draw, size, n_pockets):
    """Fewer, irregular large voids with dark, burnt edges — structural breakdown."""
    for _ in range(n_pockets):
        x = random.randint(15, size - 15)
        y = random.randint(15, size - 15)
        rx = random.randint(6, 22)
        ry = random.randint(4, 18)
        angle_pts = random.randint(5, 9)
        pts = []
        for k in range(angle_pts):
            theta = 2 * math.pi * k / angle_pts + random.uniform(-0.3, 0.3)
            dist = random.uniform(0.6, 1.0)
            pts.append((x + rx * dist * math.cos(theta),
                        y + ry * dist * math.sin(theta)))
        # dark burnt outline, lighter void interior
        fill_v = random.randint(130, 170)
        color = (fill_v, fill_v - random.randint(20, 50), fill_v - random.randint(50, 90))
        color = tuple(max(0, c) for c in color)
        draw.polygon(pts, fill=color, outline=(30, 15, 5))


# ── image generators ──────────────────────────────────────────────────────────

def gen_fresh(size=IMAGE_SIZE):
    img = Image.new("RGB", (size, size), _fresh_bg())
    draw = ImageDraw.Draw(img)
    # Subtle crumb grain texture: fine noise layer
    for _ in range(size * size // 6):
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        v = random.randint(-12, 12)
        px = img.getpixel((x, y))
        img.putpixel((x, y), tuple(max(0, min(255, c + v)) for c in px))
    _draw_fresh_pockets(draw, size, n_pockets=random.randint(28, 45))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def gen_stale(size=IMAGE_SIZE):
    img = Image.new("RGB", (size, size), _stale_bg())
    draw = ImageDraw.Draw(img)
    # Add a subtle moisture-gradient: slightly darker toward center
    for _ in range(size * size // 5):
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        v = random.randint(-8, 8)
        px = img.getpixel((x, y))
        img.putpixel((x, y), tuple(max(0, min(255, c + v)) for c in px))
    # darker central zone simulating moisture accumulation
    cx, cy = size // 2, size // 2
    for r_val in range(size // 3, 0, -8):
        alpha_v = int(20 * (1 - r_val / (size // 3)))
        draw.ellipse([cx - r_val, cy - r_val, cx + r_val, cy + r_val],
                     fill=None, outline=(max(0, _stale_bg()[0] - alpha_v),
                                         max(0, _stale_bg()[1] - alpha_v),
                                         max(0, _stale_bg()[2] - alpha_v)))
    _draw_stale_pockets(draw, size, n_pockets=random.randint(8, 18))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img


def gen_overbaked(size=IMAGE_SIZE):
    img = Image.new("RGB", (size, size), _overbaked_bg())
    draw = ImageDraw.Draw(img)
    # strong grainy texture - uneven caramelization
    for _ in range(size * size // 3):
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        v = random.randint(-25, 25)
        px = img.getpixel((x, y))
        img.putpixel((x, y), tuple(max(0, min(255, c + v)) for c in px))
    _draw_overbaked_pockets(draw, size, n_pockets=random.randint(6, 14))
    # a few bright-ish highlight patches (localized under-baked spots adjacent to over-baked)
    for _ in range(random.randint(2, 5)):
        x, y = random.randint(20, size - 20), random.randint(20, size - 20)
        r = random.randint(8, 20)
        hl = random.randint(140, 170)
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=(hl, hl - 30, hl - 70), outline=None)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def gen_broken(size=IMAGE_SIZE):
    """Two mismatched crumb zones with a visible crack line."""
    # Left zone: fresh-ish
    left = gen_fresh(size)
    # Right zone: overbaked
    right = gen_overbaked(size)
    img = Image.new("RGB", (size, size))
    # Crack position: slightly off-center
    split = int(size * random.uniform(0.40, 0.60))
    img.paste(left.crop((0, 0, split, size)), (0, 0))
    img.paste(right.crop((split, 0, size, size)), (split, 0))
    draw = ImageDraw.Draw(img)
    # Draw crack line
    for y in range(size):
        jitter = random.randint(-3, 3)
        x = split + jitter
        draw.line([(x - 1, y), (x + 1, y)], fill=(20, 12, 5), width=2)
    return img


GENERATORS = {
    "fresh": gen_fresh,
    "stale": gen_stale,
    "overbaked": gen_overbaked,
    "broken": gen_broken,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=15, help="Images per class")
    parser.add_argument(
        "--classes", nargs="+",
        default=["fresh", "stale", "overbaked"],
        choices=list(GENERATORS.keys()),
    )
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    print("=== Generating SYNTHETIC vision dataset (pipeline testing only) ===\n")
    print("*** Replace with real cross-section photos before your demo! ***\n")

    for cls in args.classes:
        cls_dir = os.path.join(IMAGE_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        gen_fn = GENERATORS[cls]
        for i in range(1, args.count + 1):
            _rng_seed(SEED + hash(cls), i)
            img = gen_fn(size=args.size)
            path = os.path.join(cls_dir, f"{cls}_{i:02d}.jpg")
            img.save(path, "JPEG", quality=92)
        print(f"  {cls:<12} -> {args.count} synthetic images in {cls_dir}/")

    print("\nDone.")
    print("Run: python train_vision.py --epochs 15")
    print("REMINDER: swap these for real cross-section photos before the live demo.")


if __name__ == "__main__":
    main()
