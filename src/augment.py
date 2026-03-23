import os
import cv2
import sys
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

SEED     = cfg["project"]["seed"]
IMG_SIZE = cfg["dataset"]["image_size"]   # 224

OUT_AUTH = Path(cfg["paths"]["synthetic_authentic"])
OUT_FAKE = Path(cfg["paths"]["synthetic_counterfeit"])
PATHS_TXT = Path("data/authentic_paths.txt")

random.seed(SEED)
np.random.seed(SEED)

OUT_AUTH.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)


def hue_shift(img: np.ndarray) -> np.ndarray:
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    shift = random.randint(15, 35) * random.choice([-1, 1])
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def gaussian_blur(img: np.ndarray) -> np.ndarray:
    sigma = round(random.uniform(1.2, 2.8), 1)
    k     = int(6 * sigma) | 1   # force odd
    return cv2.GaussianBlur(img, (k, k), sigma)


def jpeg_artifact(img: np.ndarray) -> np.ndarray:
    quality = random.randint(15, 35)
    _, buf  = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def channel_noise(img: np.ndarray) -> np.ndarray:
    std   = random.uniform(12, 28)
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def brightness_drop(img: np.ndarray) -> np.ndarray:
    factor = random.uniform(0.50, 0.78)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def geometric_warp(img: np.ndarray) -> np.ndarray:
    h, w   = img.shape[:2]
    max_d  = int(min(h, w) * 0.05)

    def jitter():
        return random.randint(-max_d, max_d)

    src = np.float32([[0, 0],     [w, 0],     [w, h],     [0, h]])
    dst = np.float32([
        [0 + jitter(), 0 + jitter()],
        [w + jitter(), 0 + jitter()],
        [w + jitter(), h + jitter()],
        [0 + jitter(), h + jitter()]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

TRANSFORMS = {
    "hue_shift":       hue_shift,
    "gaussian_blur":   gaussian_blur,
    "jpeg_artifact":   jpeg_artifact,
    "channel_noise":   channel_noise,
    "brightness_drop": brightness_drop,
    "geometric_warp":  geometric_warp,
}
T_NAMES = list(TRANSFORMS.keys())


def preprocess(img: np.ndarray) -> np.ndarray:
    """Resize to IMG_SIZE × IMG_SIZE using high-quality Lanczos interpolation."""
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)


def make_counterfeit(img: np.ndarray) -> tuple:
    """
    Apply 2–4 randomly selected transforms to one authentic image.

    Returns:
        (degraded_img, applied_transform_names)
    """
    n        = random.randint(2, 4)
    selected = random.sample(T_NAMES, k=n)
    result   = img.copy()
    for name in selected:
        result = TRANSFORMS[name](result)
    return result, selected

def run(dry_run: bool = False, limit: int = None):
    if not PATHS_TXT.exists():
        raise FileNotFoundError(
            f"\n{PATHS_TXT} not found.\n"
            "Run  python notebooks/01_dataset_inspection_FIXED.py  first."
        )

    with open(PATHS_TXT) as f:
        all_paths = [Path(l.strip()) for l in f if l.strip()]

    if dry_run:
        all_paths = all_paths[:20]
        print("🔬 DRY RUN — 20 images only")
    elif limit:
        all_paths = all_paths[:limit]

    total = len(all_paths)
    print(f"\n{'='*60}")
    print("STEP 3 — COUNTERFEIT GENERATION PIPELINE")
    print(f"{'='*60}")
    print(f"  Input images      : {total}")
    print(f"  Target image size : {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Transforms pool   : {T_NAMES}")
    print(f"  Per image         : 2–4 random transforms")
    print(f"  Auth output       : {OUT_AUTH}")
    print(f"  Fake output       : {OUT_FAKE}")
    print(f"{'='*60}\n")

    transform_usage = {t: 0 for t in T_NAMES}
    errors          = []
    saved           = 0

    for idx, src in enumerate(tqdm(all_paths, desc="Generating", unit="img", colour="green")):
        try:
            raw = cv2.imread(str(src))
            if raw is None:
                errors.append((str(src), "cv2.imread returned None"))
                continue

            if len(raw.shape) == 2:                  # grayscale
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            elif raw.shape[2] == 4:                  # BGRA 
                raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

            auth_img = preprocess(raw)
            fake_img, applied = make_counterfeit(auth_img)
            stem     = f"{idx:05d}_{src.stem}"
            auth_out = OUT_AUTH / f"{stem}.jpg"
            fake_out = OUT_FAKE / f"{stem}_fake.jpg"

            cv2.imwrite(str(auth_out), auth_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(fake_out), fake_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            for t in applied:
                transform_usage[t] += 1
            saved += 1

        except Exception as e:
            errors.append((str(src), str(e)))

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Images processed  : {saved}")
    print(f"Errors            : {len(errors)}")
    print(f"Auth saved to     : {OUT_AUTH}  ({saved} files)")
    print(f"Fake saved to     : {OUT_FAKE}  ({saved} files)")
    print(f"Total dataset     : {saved * 2} images")
    print(f"\n  Transform usage frequency:")
    for t, n in sorted(transform_usage.items(), key=lambda x: -x[1]):
        bar = '█' * int(n / max(transform_usage.values()) * 30)
        print(f"    {t:<20} {n:>5}  {bar}")

    if errors:
        print(f"\n  ⚠️  Failed files ({len(errors)}):")
        for path, reason in errors[:10]:
            print(f"    {path} — {reason}")

    print(f"\nRun inspection on synthetic data:")
    print(f"python src/verify_synthetic.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic counterfeit images")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only 20 images (quick test)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only N images")
    args = parser.parse_args()
    run(dry_run=args.dry_run, limit=args.limit)