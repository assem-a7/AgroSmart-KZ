"""
AgroSmart KZ — train_soil_model.py

EfficientNet-B0 transfer learning (torchvision pretrained) script.

Expected ImageFolder layout:
soil_images/
  Black_Soil/...
  Alluvial_Soil/...
  Arid_Soil/...
  Red_Soil/...
  Yellow_Soil/...
  Mountain_Soil/...
  Laterite_Soil/...

Output:
models/efficientnet_soil.pth
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "efficientnet_soil.pth"

EPOCHS = 20
LR = 0.001
BATCH_SIZE = 32
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REQUIRED_CLASSES = [
    "Black_Soil",
    "Alluvial_Soil",
    "Arid_Soil",
    "Red_Soil",
    "Yellow_Soil",
    "Mountain_Soil",
    "Laterite_Soil",
]

DATASET_CANDIDATES = [
    BASE_DIR / "soil_images",
    BASE_DIR / "data" / "soil_images",
]


def _is_valid_imagefolder_root(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    available = {p.name for p in root.iterdir() if p.is_dir()}
    return all(cls in available for cls in REQUIRED_CLASSES)


def _resolve_dataset_roots() -> list[Path]:
    valid_roots: list[Path] = []
    for root in DATASET_CANDIDATES:
        if _is_valid_imagefolder_root(root):
            valid_roots.append(root)

        # Legacy nested structure support
        for nested in ("Original-Dataset", "Orignal-Dataset", "CyAUG-Dataset"):
            candidate = root / nested
            if _is_valid_imagefolder_root(candidate):
                valid_roots.append(candidate)

    # unique preserve order
    uniq: list[Path] = []
    seen = set()
    for r in valid_roots:
        key = str(r.resolve())
        if key not in seen:
            uniq.append(r)
            seen.add(key)

    if not uniq:
        raise FileNotFoundError(
            "Dataset not found. Expected ImageFolder format under "
            "'soil_images/<class_name>/' or nested Original/Orignal/CyAUG-Dataset folders."
        )
    return uniq


def _build_transforms():
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def _build_dataloaders(dataset_roots: list[Path]):
    train_tf, val_tf = _build_transforms()

    train_sets = [datasets.ImageFolder(str(root), transform=train_tf) for root in dataset_roots]
    val_sets = [datasets.ImageFolder(str(root), transform=val_tf) for root in dataset_roots]

    class_names = train_sets[0].classes
    full_train = ConcatDataset(train_sets)
    full_val = ConcatDataset(val_sets)
    n = len(full_train)
    val_size = max(1, int(n * 0.2))
    train_size = n - val_size

    train_subset, val_subset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_subset = torch.utils.data.Subset(full_val, val_subset.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    return train_loader, val_loader, class_names


def _build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model.to(DEVICE)


def _run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if train_mode:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total += int(images.size(0))

    avg_loss = total_loss / max(total, 1)
    avg_acc = total_correct / max(total, 1)
    return avg_loss, avg_acc


def train():
    dataset_roots = _resolve_dataset_roots()
    train_loader, val_loader, class_names = _build_dataloaders(dataset_roots)

    model = _build_model(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training failed: best state is empty.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "class_names": class_names,
            "num_classes": len(class_names),
            "img_size": IMG_SIZE,
            "best_val_acc": float(best_acc),
            "epochs": EPOCHS,
            "lr": LR,
            "scheduler": "StepLR(step_size=7,gamma=0.5)",
            "dataset_roots": [str(p) for p in dataset_roots],
        },
        str(MODEL_PATH),
    )
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
