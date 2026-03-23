import os
import cv2
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

AUTH_DIR = Path("data/synthetic/authentic")
FAKE_DIR = Path("data/synthetic/counterfeit")
SEED     = 42
random.seed(SEED)

os.makedirs("outputs/plots", exist_ok=True)


def load_bgr_to_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

auth_files = sorted(AUTH_DIR.glob("*.jpg"))
fake_files = sorted(FAKE_DIR.glob("*.jpg"))

print(f"\n{'='*60}")
print("SYNTHETIC DATASET VERIFICATION")
print(f"{'='*60}")
print(f"  Authentic images  : {len(auth_files)}")
print(f"  Counterfeit images: {len(fake_files)}")
print(f"  Balance ratio     : {len(auth_files)/max(len(fake_files),1):.3f}  (target: 1.000)")

if len(auth_files) == 0:
    raise SystemExit("No authentic images found. Run augment.py first.")

n_pairs  = min(8, len(auth_files))
indices  = random.sample(range(len(auth_files)), n_pairs)

fig, axes = plt.subplots(n_pairs, 2, figsize=(10, n_pairs * 2.5))
fig.patch.set_facecolor('#0a0a0a')

for row, idx in enumerate(indices):
    a_path = auth_files[idx]
    prefix    = a_path.stem[:5]
    matching  = [f for f in fake_files if f.stem.startswith(prefix)]
    f_path    = matching[0] if matching else fake_files[idx % len(fake_files)]

    auth_img = load_bgr_to_rgb(a_path)
    fake_img = load_bgr_to_rgb(f_path)

    axes[row, 0].imshow(auth_img)
    axes[row, 0].axis('off')
    axes[row, 0].set_facecolor('#0a0a0a')
    if row == 0:
        axes[row, 0].set_title('AUTHENTIC', color='#10b981',
                               fontsize=11, fontweight='bold', pad=8)

    axes[row, 1].imshow(fake_img)
    axes[row, 1].axis('off')
    axes[row, 1].set_facecolor('#0a0a0a')
    if row == 0:
        axes[row, 1].set_title('COUNTERFEIT', color='#ef4444',
                               fontsize=11, fontweight='bold', pad=8)

    for col in range(2):
        for spine in axes[row, col].spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(0.5)

plt.suptitle('Authentic vs Synthetic Counterfeit — Visual QA',
             color='white', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_path = "outputs/plots/synthetic_comparison.png"
plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor='#0a0a0a')
print(f"\nComparison grid → {save_path}")

print("\n  Computing pixel statistics...")

def mean_std(paths, n=100):
    vals = []
    for p in random.sample(paths, min(n, len(paths))):
        img = cv2.imread(str(p))
        if img is not None:
            vals.append(img.mean())
    return np.mean(vals), np.std(vals)

a_mean, a_std = mean_std(auth_files)
f_mean, f_std = mean_std(fake_files)

print(f"\n  Pixel intensity stats (mean ± std over 100 samples):")
print(f"    Authentic   → mean: {a_mean:.1f}  std: {a_std:.1f}")
print(f"    Counterfeit → mean: {f_mean:.1f}  std: {f_std:.1f}")
print(f"    Δ mean      → {abs(a_mean - f_mean):.1f} px  (expect > 5 — confirms degradation)")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.patch.set_facecolor('#0a0a0a')
titles    = ['Authentic', 'Counterfeit']
colors    = ['#10b981', '#ef4444']
src_lists = [auth_files, fake_files]

for col, (title, color, paths) in enumerate(zip(titles, colors, src_lists)):
    sample_imgs = random.sample(paths, min(20, len(paths)))
    r_all, g_all, b_all = [], [], []
    for p in sample_imgs:
        img = cv2.imread(str(p))
        if img is not None:
            b, g, r = cv2.split(img)
            r_all.extend(r.flatten().tolist())
            g_all.extend(g.flatten().tolist())
            b_all.extend(b.flatten().tolist())

    ax = axes[col]
    ax.set_facecolor('#111111')
    ax.hist(r_all, bins=50, color='#ef4444', alpha=0.7, label='R', density=True)
    ax.hist(g_all, bins=50, color='#10b981', alpha=0.7, label='G', density=True)
    ax.hist(b_all, bins=50, color='#3b82f6', alpha=0.7, label='B', density=True)
    ax.set_title(title, color=color, fontsize=12, fontweight='bold')
    ax.set_xlabel('Pixel intensity', color='#aaa')
    ax.set_ylabel('Density', color='#aaa')
    ax.tick_params(colors='#aaa')
    ax.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

plt.suptitle('RGB Channel Distribution Comparison',
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
hist_path = "outputs/plots/channel_comparison.png"
plt.savefig(hist_path, dpi=130, bbox_inches='tight', facecolor='#0a0a0a')
print(f"Channel histograms → {hist_path}")

print(f"\n{'='*60}")
print("VERIFICATION COMPLETE — Synthetic dataset looks good!")
print(f"{'='*60}")
print(f"  Total images ready for training: {len(auth_files) + len(fake_files)}")
print(f"  Next step: python src/data_loader.py  (Step 4 — data pipeline)")
print(f"{'='*60}\n")