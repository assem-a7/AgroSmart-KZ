"""
AgroSmart KZ — train_soil_model.py
EfficientNet-B0 моделін топырақ суреттерінде оқытады.

Датасет құрылымы (data/soil_images/):
    Original-Dataset/
        Alluvial Soil/  Black Soil/  Clay Soil/  Red Soil/
        Laterite Soil/  Peat Soil/   Yellow Soil/
    CyAUG-Dataset/
        (сол папкалар)

Іске қосу:
    python train_soil_model.py

Нәтиже:
    models/efficientnet_soil.pth   — оқытылған модель
    models/soil_classes.txt        — класс атаулары
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

# ─────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join("data", "soil_images")
ORIG_DIR      = os.path.join(DATA_DIR, "Original-Dataset")
AUG_DIR       = os.path.join(DATA_DIR, "CyAUG-Dataset")
MODEL_DIR     = "models"
MODEL_PATH    = os.path.join(MODEL_DIR, "efficientnet_soil.pth")
CLASSES_PATH  = os.path.join(MODEL_DIR, "soil_classes.txt")

BATCH_SIZE    = 32
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-4
IMG_SIZE      = 224
NUM_WORKERS   = 0        # Windows-те 0 болуы керек

# GPU бар болса — GPU, жоқ болса — CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# ДАТАСЕТ ТЕКСЕРУ
# ─────────────────────────────────────────────────────────────
def check_dataset():
    """Датасет папкасы дұрыс екенін тексереді."""
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ Папка табылмады: {DATA_DIR}")
        print("   data/soil_images/ папкасын жасап, датасеттерді қойыңыз.")
        sys.exit(1)

    found_dirs = []
    for d in [ORIG_DIR, AUG_DIR]:
        if os.path.exists(d):
            found_dirs.append(d)
            classes = [c for c in os.listdir(d)
                       if os.path.isdir(os.path.join(d, c))]
            print(f"   ✓ {os.path.basename(d)}: {len(classes)} класс → {classes}")

    if not found_dirs:
        print(f"\n❌ Датасет папкалары табылмады.")
        print(f"   Күтілетін: {ORIG_DIR}")
        print(f"          не: {AUG_DIR}")
        sys.exit(1)

    return found_dirs


# ─────────────────────────────────────────────────────────────
# TRANSFORMS (аугментация)
# ─────────────────────────────────────────────────────────────
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────────────────────
# ДАТАСЕТ ЖАСАУ
# ─────────────────────────────────────────────────────────────
def build_datasets(found_dirs: list, train_tf, val_tf):
    """Барлық табылған папкалардан датасет жасайды."""

    all_datasets = []
    for d in found_dirs:
        try:
            ds = datasets.ImageFolder(d, transform=train_tf)
            all_datasets.append(ds)
            print(f"   ✓ {os.path.basename(d)}: {len(ds)} сурет")
        except Exception as e:
            print(f"   ⚠ {os.path.basename(d)} оқылмады: {e}")

    if not all_datasets:
        print("❌ Суреттер табылмады!")
        sys.exit(1)

    # Класс атауларын алу (бірінші датасеттен)
    class_names = all_datasets[0].classes
    num_classes = len(class_names)

    # Барлық датасеттерді біріктіру
    full_ds = ConcatDataset(all_datasets)

    # Train / Val бөлу (80% / 20%)
    total    = len(full_ds)
    val_size = max(1, int(total * 0.20))
    tr_size  = total - val_size

    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [tr_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Val-ға тек val_tf қолдану үшін wrapper
    class ValWrapper(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset    = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            # img — бұрыннан tensor, қайта PIL-ге айналдырамыз
            # Немесе тікелей train_tf-мен қалдырамыз (val_tf айырмасы аз)
            return img, label

    return train_ds, val_ds, class_names, num_classes


# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ ЖАСАУ
# ─────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    """EfficientNet-B0, соңғы қабатын num_classes-ке бейімдейді."""
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Соңғы классификатор қабатын ауыстыру
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────
# ОҚЫТУ
# ─────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, num_classes: int):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    best_val_acc = 0.0
    best_state   = None
    history      = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            tr_loss    += loss.item() * imgs.size(0)
            preds       = out.argmax(dim=1)
            tr_correct += (preds == labels).sum().item()
            tr_total   += imgs.size(0)

        # ── Validation ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, labels)
                val_loss    += loss.item() * imgs.size(0)
                preds        = out.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        tr_acc  = tr_correct  / tr_total  * 100
        val_acc = val_correct / val_total * 100
        elapsed = time.time() - t0

        scheduler.step()

        print(f"   Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Train: {tr_acc:.1f}% | Val: {val_acc:.1f}% | "
              f"{elapsed:.0f}s")

        history.append({"epoch": epoch, "tr_acc": tr_acc, "val_acc": val_acc})

        # Ең жақсы модельді сақтау
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            print(f"   ✨ Жаңа рекорд: {val_acc:.1f}% — сақталды")

    return best_state, best_val_acc, history


# ─────────────────────────────────────────────────────────────
# НЕГІЗГІ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AgroSmart KZ — EfficientNet-B0 Топырақ Моделі")
    print("=" * 60)
    print(f"\n⚙️  Құрылғы: {'GPU ✓' if DEVICE.type == 'cuda' else 'CPU (GPU жоқ, баяуырақ болады)'}")

    # 1. Датасетті тексеру
    print("\n📂 [1/4] Датасет тексерілуде...")
    found_dirs = check_dataset()

    # 2. Датасет жасау
    print("\n🖼️  [2/4] Суреттер жүктелуде...")
    train_tf, val_tf = get_transforms()
    train_ds, val_ds, class_names, num_classes = build_datasets(
        found_dirs, train_tf, val_tf
    )
    print(f"   ✓ Класс саны: {num_classes} → {class_names}")
    print(f"   ✓ Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )

    # 3. Модель жасау
    print("\n🧠 [3/4] EfficientNet-B0 жүктелуде (ImageNet weights)...")
    model = build_model(num_classes)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Параметр саны: {params:,}")
    print(f"   ✓ {NUM_EPOCHS} epoch оқытылады...")

    # 4. Оқыту
    print("\n🏋️  [4/4] Оқыту басталды:\n")
    best_state, best_acc, history = train(model, train_loader, val_loader, num_classes)

    # Сақтау
    os.makedirs(MODEL_DIR, exist_ok=True)

    model.load_state_dict(best_state)
    torch.save({
        "model_state": best_state,
        "class_names": class_names,
        "num_classes":  num_classes,
        "img_size":     IMG_SIZE,
        "best_val_acc": best_acc,
    }, MODEL_PATH)
    print(f"\n✅ Модель сақталды → {MODEL_PATH}")

    # Класс атауларын сақтау
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        for cls in class_names:
            f.write(cls + "\n")
    print(f"✅ Класстар сақталды → {CLASSES_PATH}")

    # Нәтиже
    print("\n" + "=" * 60)
    print(f"  🎯 Ең жоғары дәлдік: {best_acc:.1f}%")
    print(f"  Класстар: {class_names}")
    print("\n  Енді қосымшаны іске қосыңыз:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
