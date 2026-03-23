import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml


with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED       = cfg["project"]["seed"]
IMG_SIZE   = cfg["dataset"]["image_size"]
BATCH_SIZE = cfg["dataset"]["batch_size"]
WORKERS    = cfg["dataset"]["num_workers"]
VAL_SPLIT  = cfg["dataset"]["val_split"]
TEST_SPLIT = cfg["dataset"]["test_split"]
MAX_EACH   = cfg["dataset"]["max_samples_per_class"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

AUTH_DIR = Path(cfg["paths"]["synthetic_authentic"])
FAKE_DIR = Path(cfg["paths"]["synthetic_counterfeit"])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                   saturation=0.10, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ])

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MedicineDataset(Dataset):

    def __init__(self, files: list, labels: list, transform=None):
        self.files     = files
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img   = Image.open(self.files[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def build_file_list():

    auth_all = sorted(AUTH_DIR.glob("*.jpg"))
    fake_all = sorted(FAKE_DIR.glob("*.jpg"))

    if not auth_all:
        raise FileNotFoundError(f"No files in {AUTH_DIR}. Run src/augment.py first.")
    if not fake_all:
        raise FileNotFoundError(f"No files in {FAKE_DIR}. Run src/augment.py first.")

    rng = random.Random(SEED)
    auth_paths = rng.sample(auth_all, min(MAX_EACH, len(auth_all)))
    fake_paths = rng.sample(fake_all, min(MAX_EACH, len(fake_all)))

    paths  = [str(p) for p in auth_paths] + [str(p) for p in fake_paths]
    labels = [0] * len(auth_paths) + [1] * len(fake_paths)

    return paths, labels


def get_dataloaders():

    paths, labels = build_file_list()

    tr_val_p, te_p, tr_val_l, te_l = train_test_split(
        paths, labels, test_size=TEST_SPLIT,
        random_state=SEED, stratify=labels
    )

    relative_val = VAL_SPLIT / (1.0 - TEST_SPLIT)

    tr_p, va_p, tr_l, va_l = train_test_split(
        tr_val_p, tr_val_l, test_size=relative_val,
        random_state=SEED, stratify=tr_val_l
    )

    train_ds = MedicineDataset(tr_p, tr_l, get_transforms("train"))
    val_ds   = MedicineDataset(va_p, va_l, get_transforms("val"))
    test_ds  = MedicineDataset(te_p, te_l, get_transforms("test"))

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        pin_memory=False,
        persistent_workers=WORKERS > 0
    )

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print("\n" + "="*60)
    print("DATA PIPELINE READY")
    print("="*60)
    print(f"  Total images    : {len(paths)}")
    print(f"  Train           : {len(train_ds)}  ({len(train_loader)} batches)")
    print(f"  Val             : {len(val_ds)}  ({len(val_loader)} batches)")
    print(f"  Test            : {len(test_ds)}  ({len(test_loader)} batches)")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Workers         : {WORKERS}")
    print(f"  Image size      : {IMG_SIZE}x{IMG_SIZE}")
    print("="*60)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape : {images.shape}")
    print(f"Label shape : {labels.shape}")
    print(f"Pixel range : [{images.min():.3f}, {images.max():.3f}]")
    print("data_loader.py verified.")