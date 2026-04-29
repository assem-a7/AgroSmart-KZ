"""
# AgroSmart-KZ v2.1 — 2026, Decentrathon 5.0
"""
"""
AgroSmart KZ — soil_analyzer.py v3.0
EfficientNet-B0 → топырақ типін анықтау
YOLOv8         → проблема детекциясы
Crop CSV       → егін ұсынысы
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms, models
    from torchvision.models import EfficientNet_B0_Weights
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    F = None
    transforms = None
    models = None
    EfficientNet_B0_Weights = None
    nn = None
    _TORCH_AVAILABLE = False

from config import REGION_CROPS

# ─────────────────────────────────────────────────────────────
# Жолдар
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "efficientnet_soil.pth")
CROP_CSV   = os.path.join("data", "Crop_recommendation.csv")
DEVICE     = torch.device("cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if _TORCH_AVAILABLE else None
IMG_SIZE   = 224

# ─────────────────────────────────────────────────────────────
# Топырақ типтері — датасет класстарымен ДӘЛМЕ-ДӘЛ сәйкес
# (астын сызықты да, бос орынды да қабылдайды)
# ─────────────────────────────────────────────────────────────
SOIL_INFO = {
    "Alluvial Soil": {
        "kz": "Аллювиалды топырақ", "rating": 5,
        "ru": "Аллювиальная — очень плодородная, речные отложения",
        "recs": [
            "✅ Егін егуге өте қолайлы",
            "💧 Орташа суару жеткілікті",
            "🌾 Бидай, жүгері, күріш өте жақсы өседі",
            "🌱 Азоттық тыңайтқыш аз қажет",
        ],
    },
    "Arid Soil": {
        "kz": "Құрғақ (Аридті) топырақ", "rating": 2,
        "ru": "Аридная — сухая, пустынная, бедная питательными веществами",
        "recs": [
            "❌ Суаруссыз егін егу мүмкін емес",
            "💧 Тамшылатып суару жүйесі міндетті",
            "🐑 Мал жайылымы ретінде ғана жарамды",
            "🌱 Органикалық тыңайтқыш + топырақ жақсартқыш керек",
        ],
    },
    "Black Soil": {
        "kz": "Қара топырақ (Чернозём)", "rating": 5,
        "ru": "Чернозём — самый плодородный тип Казахстана",
        "recs": [
            "✅ Қазақстанның ең құнарлы топырағы",
            "💧 Орташа суару жеткілікті",
            "🌾 Бидай, арпа, күнбағыс тамаша өседі",
            "🌱 Минималды тыңайтқыш керек",
        ],
    },
    "Laterite Soil": {
        "kz": "Латеритті топырақ", "rating": 2,
        "ru": "Латеритная — бедная, кислая, выщелоченная",
        "recs": [
            "⚠️ Органикалық заттар жеткіліксіз",
            "🌱 Әк (известь) қосып pH жақсарту керек",
            "💧 Жиі суару міндетті",
            "🔧 Арнайы дайындықсыз пайдалану қиын",
        ],
    },
    "Mountain Soil": {
        "kz": "Таулы топырақ", "rating": 2,
        "ru": "Горная — каменистая, маломощная, крутые склоны",
        "recs": [
            "⚠️ Тастар көп, жырту қиын",
            "🌿 Жем-шөп дақылдары ғана өседі",
            "🐑 Мал жайылымы үшін жарамды",
            "🔧 Арнайы ауыр техника қажет",
        ],
    },
    "Red Soil": {
        "kz": "Қызыл топырақ", "rating": 3,
        "ru": "Красная — бедна азотом, требует удобрений",
        "recs": [
            "⚠️ Азоттық тыңайтқыш міндетті",
            "💧 Тамшылатып суару ұсынылады",
            "🌱 pH деңгейін тексеру керек",
            "🌾 Бақша дақылдары жақсы өседі",
        ],
    },
    "Yellow Soil": {
        "kz": "Сарғыш топырақ", "rating": 3,
        "ru": "Жёлтая — умеренная, требует органики",
        "recs": [
            "⚠️ Органикалық тыңайтқыш міндетті",
            "💧 Тұрақты суару ұсынылады",
            "🌾 Дәнді дақылдарға жарамды",
            "🌱 Топырақ жақсартқышпен нәтиже артады",
        ],
    },
    # Clay Soil — кейбір датасеттерде болады
    "Clay Soil": {
        "kz": "Сазды топырақ", "rating": 3,
        "ru": "Глинистая — тяжёлая, требует дренажа",
        "recs": [
            "⚠️ Дренаж жүйесі міндетті",
            "🔧 Кесектеу (рыхление) жылына 2 рет",
            "💧 Артық суды бақылау керек",
            "🌿 Органикалық тыңайтқыш қосу ұсынылады",
        ],
    },
}

# Артқы нұсқа (белгісіз класс үшін)
DEFAULT_INFO = {
    "kz": "Белгісіз топырақ", "ru": "Тип не определён",
    "rating": 2,
    "recs": ["⚠️ Топырақ типі анықталмады", "🔬 Зертханалық анализ ұсынылады"],
}

# SOIL_INFO кілттерін іздеу функциясы
def _get_soil_info(soil_type: str) -> dict:
    """
    Астын сызық (_) немесе бос орын ( ) болса да дұрыс табады.
    Мысалы: 'Black_Soil' → 'Black Soil' → SOIL_INFO-дан алады.
    """
    # 1. Тікелей іздеу
    if soil_type in SOIL_INFO:
        return SOIL_INFO[soil_type]
    # 2. Астын сызықты бос орынға ауыстырып іздеу
    normalized = soil_type.replace("_", " ")
    if normalized in SOIL_INFO:
        return SOIL_INFO[normalized]
    # 3. Регистрсіз іздеу
    for key in SOIL_INFO:
        if key.lower() == normalized.lower():
            return SOIL_INFO[key]
    return DEFAULT_INFO


# Егін ұсынысы үшін
SOIL_PARAMS = {
    "Black Soil":    {"N": 80, "P": 50, "K": 40, "ph": 7.0},
    "Alluvial Soil": {"N": 90, "P": 60, "K": 50, "ph": 7.2},
    "Arid Soil":     {"N": 20, "P": 15, "K": 10, "ph": 7.5},
    "Clay Soil":     {"N": 60, "P": 40, "K": 60, "ph": 6.5},
    "Red Soil":      {"N": 50, "P": 35, "K": 30, "ph": 6.2},
    "Laterite Soil": {"N": 30, "P": 20, "K": 20, "ph": 5.5},
    "Mountain Soil": {"N": 35, "P": 22, "K": 18, "ph": 6.0},
    "Peat Soil":     {"N": 40, "P": 25, "K": 15, "ph": 5.0},
    "Yellow Soil":   {"N": 55, "P": 38, "K": 35, "ph": 6.8},
}

CROP_MAP = {
    "rice": "Күріш", "maize": "Жүгері", "wheat": "Бидай",
    "barley": "Арпа", "cotton": "Мақта", "sunflower": "Күнбағыс",
    "jute": "Зығыр", "lentil": "Жасымық", "mungbean": "Мұнбұршақ",
    "blackgram": "Қара бұршақ", "chickpea": "Бұршақ",
    "kidneybeans": "Бұршақ", "pomegranate": "Анар",
    "mango": "Манго", "grapes": "Жүзім", "watermelon": "Қарбыз",
    "muskmelon": "Қауын", "apple": "Алма", "orange": "Апельсин",
    "papaya": "Папайя", "coconut": "Кокос", "banana": "Банан",
    "coffee": "Кофе",
}

# ─────────────────────────────────────────────────────────────
# Жаһандық нысандар
# ─────────────────────────────────────────────────────────────
_efficientnet = None
_class_names  = None
_yolo_model   = None
_crop_df      = None


def _load_efficientnet() -> bool:
    global _efficientnet, _class_names
    if _efficientnet is not None:
        return True
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError("Checkpoint форматы dict болуы керек")

        model_state = ckpt.get("model_state") or ckpt.get("state_dict")
        class_names = ckpt.get("class_names")
        num_classes = int(ckpt.get("num_classes", 0))

        if model_state is None:
            raise ValueError("Checkpoint ішінде model_state/state_dict жоқ")

        if not class_names and num_classes > 0:
            class_names = [f"class_{i}" for i in range(num_classes)]
        if not class_names:
            raise ValueError("Checkpoint ішінде class_names жоқ")

        if num_classes <= 0:
            num_classes = len(class_names)

        _class_names = list(class_names)

        model = models.efficientnet_b0(weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_feat, num_classes),
        )
        model.load_state_dict(model_state)
        model.eval().to(DEVICE)
        _efficientnet = model
        return True
    except Exception as e:
        print(f"⚠ EfficientNet жүктелмеді: {e}")
        return False


def _load_yolo() -> bool:
    global _yolo_model
    if _yolo_model is not None:
        return True
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
        return True
    except Exception as e:
        print(f"⚠ YOLO жүктелмеді: {e}")
        return False


def _load_crop_df():
    global _crop_df
    if _crop_df is not None:
        return _crop_df
    if os.path.exists(CROP_CSV):
        try:
            _crop_df = pd.read_csv(CROP_CSV)
        except Exception:
            pass
    return _crop_df


# ─────────────────────────────────────────────────────────────
# EfficientNet классификациясы
# ─────────────────────────────────────────────────────────────
def _detect_cracks_simple(images: list) -> float:
    scores = []
    for img in images:
        arr  = np.array(img.convert("L").resize((224, 224)), dtype=np.float32)
        mean = arr.mean()
        gx   = np.abs(np.diff(arr, axis=1)).mean()
        gy   = np.abs(np.diff(arr, axis=0)).mean()
        grad = (gx + gy) / 2
        crack_sig = 0.0
        if 70 < mean < 175 and grad > 7:
            crack_sig = min(1.0, grad / 18.0)
        scores.append(crack_sig)
    return float(np.mean(scores))


def _classify_efficientnet(images: list) -> tuple:
    TEMPERATURE = 0.4

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    all_logits = []
    with torch.no_grad():
        for img in images:
            t      = tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
            logits = _efficientnet(t).cpu().numpy()[0]
            all_logits.append(logits)

    avg_logits = np.mean(all_logits, axis=0)
    scaled     = avg_logits / TEMPERATURE
    exp        = np.exp(scaled - scaled.max())
    avg        = exp / exp.sum()

    crack_score = _detect_cracks_simple(images)
    arid_idx    = next((i for i, c in enumerate(_class_names)
                        if "arid" in c.lower()), None)
    if crack_score > 0.25 and arid_idx is not None:
        avg[arid_idx] = max(avg[arid_idx], crack_score * 0.85)
        avg = avg / avg.sum()

    best      = int(np.argmax(avg))
    conf      = round(float(avg[best]) * 100, 1)
    scores    = {
        _class_names[i].replace("_", " "): round(float(avg[i]) * 100, 1)
        for i in range(len(_class_names))
    }
    soil_type = _class_names[best].replace("_", " ")
    return soil_type, conf, scores


# Fallback — PIL түс анализі
def _classify_fallback(images: list) -> tuple:
    feats = []
    for img in images:
        a = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
        r, g, b = a[:,:,0].mean(), a[:,:,1].mean(), a[:,:,2].mean()
        feats.append({"r": r, "g": g, "b": b,
                      "dark": 1 - (r+g+b)/3/255})
    avg = {k: float(np.mean([f[k] for f in feats])) for k in feats[0]}

    scores = {
        "Black Soil":    min(100, avg["dark"] * 150),
        "Alluvial Soil": min(100, max(0, (avg["g"] - avg["b"]) * 2 + 30)),
        "Arid Soil":     min(100, max(0, avg["r"] * 0.4)),
        "Red Soil":      min(100, max(0, (avg["r"] - avg["g"]) * 2)),
        "Mountain Soil": min(100, max(0, (avg["r"] + avg["b"] - avg["g"]) * 0.5)),
        "Laterite Soil": 15.0,
        "Yellow Soil":   min(100, max(0, (avg["r"] + avg["g"] - avg["b"]) * 0.5)),
    }
    best  = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    conf  = round(scores[best] / total * 100, 1)
    norm  = {k: round(v / total * 100, 1) for k, v in scores.items()}
    return best, conf, norm


# ─────────────────────────────────────────────────────────────
# YOLO детекция
# ─────────────────────────────────────────────────────────────
def _detect_problems(images: list) -> dict:
    problems = {}
    if not _load_yolo():
        return problems

    crack_cnt = 0
    stone_cnt = 0
    dry_cnt   = 0

    try:
        for img in images:
            img_np  = np.array(img.convert("RGB"))
            results = _yolo_model(img_np, verbose=False, conf=0.25)
            for r in results:
                if r.boxes is None:
                    continue
                for cls_id in r.boxes.cls.cpu().numpy():
                    name = _yolo_model.names.get(int(cls_id), "").lower()
                    if any(k in name for k in ["crack", "fracture"]):
                        crack_cnt += 1
                    elif any(k in name for k in ["stone", "rock", "gravel"]):
                        stone_cnt += 1

            # Жасылдық индексі
            arr = np.array(img.convert("RGB"), dtype=np.float32)
            g_ratio = (arr[:,:,1] > arr[:,:,0] * 1.1).mean()
            if g_ratio < 0.05:
                dry_cnt += 1

    except Exception as e:
        print(f"⚠ YOLO қатесі: {e}")

    if crack_cnt > 0:
        problems["🔴 Жерде жарылулар бар"] = crack_cnt
    if stone_cnt > 0:
        problems["🪨 Тастар анықталды"] = stone_cnt
    if dry_cnt >= max(1, len(images) // 2 + 1):
        problems["🏜️ Өсімдік жоқ / Құрғақ жер"] = dry_cnt

    return problems


# ─────────────────────────────────────────────────────────────
# Егін ұсынысы
# ─────────────────────────────────────────────────────────────
def _recommend_crops(soil_type: str, region: str) -> list:
    region_crops = REGION_CROPS.get(region, [])
    normalized   = soil_type.replace("_", " ")

    df = _load_crop_df()
    csv_crops = []
    if df is not None:
        params = SOIL_PARAMS.get(normalized,
                 SOIL_PARAMS.get(soil_type,
                 SOIL_PARAMS["Black Soil"]))
        try:
            df2 = df.copy()
            df2["_score"] = (
                (df2["N"] - params["N"]).abs() +
                (df2["P"] - params["P"]).abs() +
                (df2["K"] - params["K"]).abs() +
                (df2["ph"] - params["ph"]).abs() * 10
            )
            top = df2.nsmallest(5, "_score")["label"].tolist()
            csv_crops = [CROP_MAP.get(c.lower(), c.capitalize()) for c in top]
        except Exception:
            pass

    combined = list(dict.fromkeys(region_crops + csv_crops))
    return combined[:6] if combined else ["Бидай", "Жоңышқа", "Арпа"]


# ─────────────────────────────────────────────────────────────
# Жалпы баға
# ─────────────────────────────────────────────────────────────
def _verdict(rating: int, problems: dict) -> tuple:
    penalty = sum([
        1.0 if "Жарылулар" in k else
        0.5 if "Тастар"    in k else
        0.5 if "Құрғақ"    in k else 0.0
        for k in problems
    ])
    eff = max(0.0, rating - penalty)

    if eff >= 4:
        return "ЖАРАМДЫ ✅", "#27ae60", "Жер егін егуге толық қолайлы"
    elif eff >= 2.5:
        return "ШАРТТЫ ЖАРАМДЫ ⚠️", "#f39c12", "Дайындаудан кейін пайдалануға болады"
    else:
        return "ЖАРАМСЫЗ ❌", "#e74c3c", "Дәнді дақылдарға жарамсыз, дайындық қажет"


# ─────────────────────────────────────────────────────────────
# НЕГІЗГІ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────
def analyze_photos(images: list, region: str) -> dict:
    if not images:
        return {"error": "Фотолар жоқ / Нет фотографий"}

    # 1. Топырақ классификациясы
    if _load_efficientnet():
        soil_type, confidence, all_scores = _classify_efficientnet(images)
        model_used = f"EfficientNet-B0 ✓ ({confidence:.0f}% сенімділік)"
    else:
        soil_type, confidence, all_scores = _classify_fallback(images)
        model_used = "Түс анализі (EfficientNet оқытылмаған)"

    # 2. Топырақ ақпараты
    info = _get_soil_info(soil_type)

    # 3. YOLO детекция
    problems = _detect_problems(images)

    # 4. Егін ұсынысы
    crops = _recommend_crops(soil_type, region)

    # 5. Жалпы баға
    verdict, verdict_color, verdict_ru = _verdict(info["rating"], problems)

    # 6. Фото детальдары
    photo_details = []
    for i, img in enumerate(images):
        arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
        brightness = arr.mean() / 255 * 100
        photo_details.append({
            "label":      f"Фото {i + 1}",
            "brightness": round(brightness, 1),
            "darkness":   round(100 - brightness, 1),
            "texture":    round(float(arr.std()), 1),
        })

    dark_vals   = [p["darkness"] for p in photo_details]
    consistency = max(0.0, 1.0 - float(np.std(dark_vals)) /
                      (float(np.mean(dark_vals)) + 1e-6)) * 100

    return {
        "soil_type":         soil_type,
        "soil_kz":           info.get("kz", soil_type),
        "soil_ru":           info.get("ru", ""),
        "rating":            info["rating"],
        "confidence":        confidence,
        "all_scores":        all_scores,
        "verdict":           verdict,
        "verdict_color":     verdict_color,
        "verdict_ru":        verdict_ru,
        "problems":          problems,
        "problems_found":    len(problems) > 0,
        "recommended_crops": crops,
        "recommendations":   info.get("recs", []),
        "region":            region,
        "n_photos":          len(images),
        "consistency":       round(min(consistency, 100.0), 1),
        "photo_details":     photo_details,
        "model_used":        model_used,
    }
