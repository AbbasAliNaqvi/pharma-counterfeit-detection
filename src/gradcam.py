import sys
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model import build_model


with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH  = Path(cfg["paths"]["models"]) / "best_model.pth"
OUT_GRADCAM = Path(cfg["paths"]["outputs"]) / "gradcam"
OUT_INDIV   = OUT_GRADCAM / "individual"
CLASS_NAMES = cfg["classes"]
SEED        = cfg["project"]["seed"]
IMG_SIZE    = cfg["dataset"]["image_size"]

OUT_GRADCAM.mkdir(parents=True, exist_ok=True)
OUT_INDIV.mkdir(parents=True, exist_ok=True)

random.seed(SEED)


class GradCAM:

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)

        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:

        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:

    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img  = tensor.squeeze(0).cpu().numpy()
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    img  = (img * 255).astype(np.uint8)
    return img.transpose(1, 2, 0)


def overlay_heatmap(rgb_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_rgb + (1 - alpha) * rgb_img).astype(np.uint8)
    return overlay


def run(n_samples: int = 12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, device = build_model(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    target_layer = model.features[12]
    gradcam = GradCAM(model, target_layer)

    _, _, test_loader = get_dataloaders()

    collected = []
    for images, labels in test_loader:
        for i in range(images.size(0)):
            collected.append((images[i], labels[i].item()))
        if len(collected) >= n_samples * 4:
            break

    random.shuffle(collected)
    samples = collected[:n_samples]

    print(f"\n{'='*60}")
    print("GRAD-CAM GENERATION")
    print(f"{'='*60}")
    print(f"  Target layer  : features[12] (last MobileNetV3 block)")
    print(f"  Samples       : {n_samples}")
    print(f"  Output dir    : {OUT_GRADCAM}")
    print(f"{'='*60}\n")

    results = []

    for idx, (img_tensor, true_label) in enumerate(samples):
        inp = img_tensor.unsqueeze(0).to(device)
        inp.requires_grad_(False)

        with torch.enable_grad():
            heatmap, pred_label = gradcam.generate(inp)

        rgb_img = tensor_to_rgb(img_tensor)
        overlay = overlay_heatmap(rgb_img, heatmap)

        correct     = pred_label == true_label
        true_name   = CLASS_NAMES[true_label]
        pred_name   = CLASS_NAMES[pred_label]

        results.append({
            "rgb":      rgb_img,
            "heatmap":  heatmap,
            "overlay":  overlay,
            "true":     true_name,
            "pred":     pred_name,
            "correct":  correct,
            "idx":      idx
        })

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor("#0a0a0a")
        titles = ["Original", "Grad-CAM Heatmap", "Overlay"]
        imgs   = [rgb_img, plt.cm.turbo(heatmap)[:,:,:3], overlay]

        for ax, title, im in zip(axes, titles, imgs):
            ax.imshow(im)
            ax.set_title(title, color="#aaaaaa", fontsize=9)
            ax.axis("off")
            ax.set_facecolor("#0a0a0a")

        status_color = "#10b981" if correct else "#ef4444"
        status_text  = "CORRECT" if correct else "WRONG"
        fig.suptitle(
            f"True: {true_name}  |  Predicted: {pred_name}  |  {status_text}",
            color=status_color, fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(OUT_INDIV / f"sample_{idx:03d}.png",
                    dpi=110, bbox_inches="tight", facecolor="#0a0a0a")
        plt.close()

        tag = "OK" if correct else "ERR"
        print(f"  [{tag}] Sample {idx:>2}  true={true_name:<12} pred={pred_name}")

    gradcam.remove_hooks()

    n_rows = min(4, len(results))
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, n_rows * 3.5))
    fig.patch.set_facecolor("#0a0a0a")

    col_titles = ["Original", "Grad-CAM Activation", "Heatmap Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)

    for row, res in enumerate(results[:n_rows]):
        rgb_display = res["rgb"]
        hm_display  = plt.cm.turbo(res["heatmap"])[:, :, :3]
        ov_display  = res["overlay"]

        axes[row, 0].imshow(rgb_display)
        axes[row, 1].imshow(hm_display)
        axes[row, 2].imshow(ov_display)

        for col in range(3):
            axes[row, col].axis("off")
            axes[row, col].set_facecolor("#0a0a0a")

        label_color = "#10b981" if res["correct"] else "#ef4444"
        axes[row, 0].set_ylabel(
            f"{res['true']}\n-> {res['pred']}",
            color=label_color, fontsize=8, rotation=0,
            labelpad=60, va="center"
        )

    correct_patch = mpatches.Patch(color="#10b981", label="Correct prediction")
    wrong_patch   = mpatches.Patch(color="#ef4444", label="Wrong prediction")
    fig.legend(handles=[correct_patch, wrong_patch],
               loc="lower center", ncol=2,
               facecolor="#1a1a1a", labelcolor="white",
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(
        "Grad-CAM — Regions influencing counterfeit classification",
        color="white", fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    grid_path = OUT_GRADCAM / "gradcam_grid.png"
    plt.savefig(grid_path, dpi=130, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()

    n_correct = sum(1 for r in results if r["correct"])
    print(f"\n  Grid saved -> {grid_path}")
    print(f"\n{'='*60}")
    print("GRAD-CAM COMPLETE")
    print(f"{'='*60}")
    print(f"  Samples analysed   : {len(results)}")
    print(f"  Correct            : {n_correct}/{len(results)}  ({100*n_correct/len(results):.1f}%)")
    print(f"  Individual images  : {OUT_INDIV}")
    print(f"  Grid overview      : {grid_path}")
    print(f"{'='*60}")
    print("\n  Pipeline complete. Next: python app/streamlit_app.py")


if __name__ == "__main__":
    run()