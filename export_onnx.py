import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import build_model

model, device = build_model()
ckpt = torch.load("models/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

dummy = torch.randn(1, 3, 224, 224)
Path("models").mkdir(exist_ok=True)

torch.onnx.export(
    model, dummy,
    "models/pharmaguard.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=11
)
print("Exported -> models/pharmaguard.onnx")