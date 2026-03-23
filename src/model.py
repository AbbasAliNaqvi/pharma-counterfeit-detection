import torch
import torch.nn as nn
from torchvision import models
import yaml


with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

PRETRAINED  = cfg["model"]["pretrained"]
DROPOUT     = cfg["model"]["dropout"]
NUM_CLASSES = cfg["model"]["num_classes"]


class CounterfeitDetector(nn.Module):

    def __init__(self):
        super(CounterfeitDetector, self).__init__()

        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if PRETRAINED else None
        base    = models.mobilenet_v3_small(weights=weights)

        for i, block in enumerate(base.features):
            if i < 11:
                for param in block.parameters():
                    param.requires_grad = False

        self.features = base.features
        self.avgpool  = base.avgpool

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, NUM_CLASSES)
        )

        self._init_head_weights()

    def _init_head_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.head(x)

    def get_param_stats(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


def build_model(device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CounterfeitDetector().to(device)
    trainable, total = model.get_param_stats()

    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"  Backbone         : MobileNetV3-Small (pretrained={PRETRAINED})")
    print(f"  Frozen           : feature blocks 0-10")
    print(f"  Trainable        : feature blocks 11-12 + custom head")
    print(f"  Output classes   : {NUM_CLASSES}")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}  ({100*trainable/total:.1f}%)")
    print(f"  Frozen params    : {total-trainable:,}  ({100*(total-trainable)/total:.1f}%)")
    print(f"  Device           : {device}")
    print("="*60)

    return model, device


if __name__ == "__main__":
    model, device = build_model()

    dummy = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)

    print(f"\nForward pass: input={dummy.shape}  output={out.shape}  (expected [4, 2])")
    print("model.py verified. Proceed to train.py")