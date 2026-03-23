import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

RAW_DIR  = Path("data/raw")
PILL_DIR = RAW_DIR / "pill_images"
EXTS     = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

print("\n" + "="*60)
print("RAW FOLDER TREE (dirs + image counts)")
print("="*60)
count = 0
for root, dirs, files in os.walk(PILL_DIR):
    level  = root.replace(str(PILL_DIR), '').count(os.sep)
    indent = '  ' * level
    n_imgs = sum(1 for f in files if Path(f).suffix.lower() in EXTS)
    print(f"{indent}{Path(root).name}/  [{n_imgs} images]")
    count += 1
    if count > 50:
        print("  ... (truncated)")
        break

all_paths    = []
class_counts = defaultdict(int)

for p in PILL_DIR.rglob('*'):
    if p.is_file() and p.suffix.lower() in EXTS:
        all_paths.append(p)
        class_counts[p.parent.name] += 1

print("\n" + "="*60)
print(f"IMAGES FOUND PER LEAF CLASS  (total: {len(class_counts)} classes)")
print("="*60)
for cls, cnt in sorted(class_counts.items()):
    print(f"  {cls:<45} {cnt:>5}")
print(f"\n  TOTAL: {len(all_paths)} images")

if len(all_paths) == 0:
    print("\nZero images. Run these to debug:")
    print("find data/raw/pill_images -type f | head -30")
    print("ls -laR data/raw/pill_images/ | head -40")
    raise SystemExit("Fix dataset path — see commands above.")

print("\n" + "="*60)
print("IMAGE QUALITY ANALYSIS (first 50 samples)")
print("="*60)
widths, heights, corrupt = [], [], []
for p in all_paths[:50]:
    try:
        img = Image.open(p)
        widths.append(img.width)
        heights.append(img.height)
    except Exception:
        corrupt.append(str(p))

print(f"  Width  → min:{min(widths)}  max:{max(widths)}  avg:{int(np.mean(widths))}px")
print(f"  Height → min:{min(heights)}  max:{max(heights)}  avg:{int(np.mean(heights))}px")
print(f"  Corrupt files: {len(corrupt)}")

os.makedirs("outputs/plots", exist_ok=True)
seen, samples = set(), []
for p in all_paths:
    if p.parent.name not in seen:
        seen.add(p.parent.name)
        samples.append((p.parent.name, p))

samples = samples[:15]
fig, axes = plt.subplots(3, 5, figsize=(18, 11))
fig.patch.set_facecolor('#0a0a0a')
axes = axes.flatten()

for idx, (cls, p) in enumerate(samples):
    try:
        img = Image.open(p).convert('RGB').resize((224, 224))
        axes[idx].imshow(img)
    except Exception:
        pass
    axes[idx].set_title(cls[:22], color='#10b981', fontsize=8, fontweight='bold')
    axes[idx].axis('off')

for idx in range(len(samples), 15):
    axes[idx].set_visible(False)

plt.suptitle('Authentic Medicine Samples — Dataset Overview',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/plots/sample_authentic.png",
            dpi=120, bbox_inches='tight', facecolor='#0a0a0a')
print("\n Grid → outputs/plots/sample_authentic.png")

os.makedirs("data", exist_ok=True)
with open("data/authentic_paths.txt", "w") as f:
    for p in all_paths:
        f.write(str(p) + "\n")

total = len(all_paths)
print("\n" + "="*60)
print("INSPECTION COMPLETE")
print("="*60)
print(f"  Authentic images    : {total}")
print(f"  Counterfeits to gen : {total}  (1:1 balanced)")
print(f"  Final dataset total : {total * 2}")
print(f"  Train  70%          : ~{int(total*2*0.70)}")
print(f"  Val    15%          : ~{int(total*2*0.15)}")
print(f"  Test   15%          : ~{int(total*2*0.15)}")
print(f"  Path list saved     : data/authentic_paths.txt")
print("="*60)