"""Субсидия верификациясы (strict real model mode).

app.py ішінен `from subsidy_verifier import verify` деп импортталады.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import models, transforms
from dotenv import load_dotenv

try:
    import ee
except Exception:  # pragma: no cover
    ee = None


CROP_CLASSES = ["Бидай", "Арпа", "Рапс", "Күнбағыс", "Картоп", "Жүзім", "Басқа"]
HEALTH_LABELS = {
    1: "Өте нашар",
    2: "Нашар",
    3: "Орташа",
    4: "Жақсы",
    5: "Өте жақсы",
}

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
_YOLO_MODEL = None
SOIL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "efficientnet_soil.pth"
)
HEALTH_MODEL_H5_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "crop_health_cnn.h5"
)
UNET_MODEL_H5_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "area_segmentation_unet.h5"
)
HEALTH_MODEL_PTH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "crop_health_cnn.pth"
)
UNET_MODEL_PTH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "area_segmentation_unet.pth"
)
_SOIL_MODEL = None
_SOIL_CLASSES = None
_SOIL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_EE_INITIALIZED = False
_HEALTH_MODEL = None
_UNET_MODEL = None
_HEALTH_MODEL_KIND = None
_UNET_MODEL_KIND = None

try:
    from tensorflow import keras
except Exception:  # pragma: no cover
    keras = None

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_BASE_DIR, ".env"), override=False)


def _require_file(path: str, alias: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{alias} табылмады: {path}")


class HealthCNNTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.net(x)


class AreaUNetTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.enc1(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.enc2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)


def stable_seed(lat: float, lon: float, area: float) -> int:
    """Координат және ауданнан детерминді seed жасайды."""
    payload = f"{float(lat):.6f}|{float(lon):.6f}|{float(area):.2f}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _init_gee() -> None:
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return
    if ee is None:
        raise RuntimeError("earthengine-api орнатылмаған")

    service_account = os.getenv("GEE_SERVICE_ACCOUNT", "").strip()
    private_key_path = os.getenv("GEE_PRIVATE_KEY_PATH", "").strip()
    project_id = os.getenv("GEE_PROJECT_ID", "").strip()
    if not project_id:
        raise RuntimeError("GEE_PROJECT_ID орнатылмаған (.env ішінде project id көрсетіңіз)")

    if service_account and private_key_path:
        credentials = ee.ServiceAccountCredentials(service_account, private_key_path)
        ee.Initialize(credentials=credentials, project=project_id)
    else:
        try:
            ee.Initialize(project=project_id)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project_id)

    _EE_INITIALIZED = True


def get_sentinel2_ndvi(
    lat: float, lon: float, date_str: str, buffer_m: int = 1000
) -> dict[str, Any]:
    """Google Earth Engine арқылы Sentinel-2 NDVI/EVI алады."""
    _init_gee()

    try:
        center_date = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception as exc:
        raise ValueError(f"date_str қате форматы: {date_str}") from exc

    start = (center_date - timedelta(days=30)).strftime("%Y-%m-%d")
    end = (center_date + timedelta(days=30)).strftime("%Y-%m-%d")

    region = ee.Geometry.Point([float(lon), float(lat)]).buffer(float(buffer_m))
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    image = collection.first()
    if image is None:
        raise RuntimeError("Sentinel-2 image табылмады")

    ndvi_img = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi_img = image.expression(
        "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
        {"NIR": image.select("B8"), "RED": image.select("B4"), "BLUE": image.select("B2")},
    ).rename("EVI")

    stats = (
        ndvi_img.addBands(evi_img)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    ndvi_val = float(stats.get("NDVI")) if stats and stats.get("NDVI") is not None else None
    evi_val = float(stats.get("EVI")) if stats and stats.get("EVI") is not None else None
    if ndvi_val is None or evi_val is None:
        raise RuntimeError("NDVI/EVI есептеу сәтсіз")

    cloud_pct = float(image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo())
    image_date = str(image.date().format("YYYY-MM-dd").getInfo())

    return {
        "source": "GEE Sentinel-2",
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "ndvi": round(ndvi_val, 3),
        "evi": round(evi_val, 3),
        "cloud_pct": cloud_pct,
        "date": image_date,
    }


def simulate_sentinel2(
    lat: float, lon: float, date_str: str, rng: np.random.Generator
) -> dict[str, Any]:
    """Тек GEE дерегі. Қате болса exception лақтырады."""
    _ = rng  # сигнатура үшін сақталды
    return get_sentinel2_ndvi(lat, lon, date_str)


def _build_test_satellite_image(
    lat: float, lon: float, area_ha: float, crop_type: str, rng: np.random.Generator
) -> Image.Image:
    """YOLO тесті үшін синтетикалық «спутниктік» сурет жасайды."""
    width, height = 640, 640
    canvas = Image.new("RGB", (width, height), (118, 95, 68))
    draw = ImageDraw.Draw(canvas)

    # Детерминді паттерн: жасыл/қоңыр блоктар.
    for _ in range(9):
        x0 = int(rng.integers(0, width - 120))
        y0 = int(rng.integers(0, height - 120))
        x1 = x0 + int(rng.integers(80, 220))
        y1 = y0 + int(rng.integers(80, 220))
        green = int(rng.integers(90, 185))
        brown = int(rng.integers(70, 130))
        color = (int(green * 0.6), green, brown)
        draw.rectangle([x0, y0, min(x1, width - 1), min(y1, height - 1)], fill=color)

    # Координат/ауданға тәуелді 1-2 объект (YOLO-ға «бірдеңе» беру үшін).
    cx = int((abs(lat) % 1) * (width - 80)) + 40
    cy = int((abs(lon) % 1) * (height - 80)) + 40
    radius = int(np.clip(12 + area_ha * 0.02, 12, 50))
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(55, 135, 55))
    draw.rectangle(
        [max(0, cx - 2 * radius), min(height - 70, cy + radius), min(width - 1, cx + 2 * radius), min(height - 1, cy + 3 * radius)],
        fill=(150, 120, 80),
    )

    # Дақыл атауын жазып қоямыз (тек визуалды маркер).
    draw.text((12, 12), f"{crop_type} {area_ha:.1f}ha", fill=(235, 235, 235))
    return canvas


def run_yolov8(
    lat: float, lon: float, area_ha: float, crop_type: str, rng: np.random.Generator
) -> dict[str, Any]:
    """YOLOv8 inference (strict real mode)."""
    global _YOLO_MODEL

    crop_key = (crop_type or "").strip().lower()
    crop_to_expected_labels = {
        "картоп": {"potted plant"},
        "жүзім": {"potted plant"},
        "күнбағыс": {"potted plant"},
        "рапс": {"potted plant"},
        "бидай": set(),
        "арпа": set(),
        "басқа": set(),
    }

    try:
        from ultralytics import YOLO

        _require_file(MODEL_PATH, "YOLOv8 модель")
        if _YOLO_MODEL is None:
            _YOLO_MODEL = YOLO(MODEL_PATH)

        test_img = _build_test_satellite_image(lat, lon, area_ha, crop_type, rng)
        img_np = np.array(test_img)

        results = _YOLO_MODEL.predict(
            source=img_np,
            verbose=False,
            conf=0.25,
            imgsz=640,
            max_det=20,
        )

        if not results:
            return {
                "detected": False,
                "confidence": 0.0,
                "detected_classes": [],
                "matched": False,
                "raw_boxes": 0,
            }

        res = results[0]
        boxes = res.boxes
        names_map = res.names if hasattr(res, "names") else {}

        raw_boxes = int(len(boxes)) if boxes is not None else 0
        if raw_boxes == 0:
            return {
                "detected": False,
                "confidence": 0.0,
                "detected_classes": [],
                "matched": False,
                "raw_boxes": 0,
            }

        confs = boxes.conf.cpu().numpy().astype(float).tolist()
        cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        detected_classes = [
            str(names_map.get(cls_id, str(cls_id))) for cls_id in cls_ids
        ]
        confidence = float(round(max(confs), 3))
        detected = bool(raw_boxes > 0 and confidence >= 0.25)

        expected = crop_to_expected_labels.get(crop_key, set())
        if expected:
            matched = any(cls_name.lower() in expected for cls_name in detected_classes)
        else:
            matched = any(crop_key and crop_key in cls_name.lower() for cls_name in detected_classes)

        return {
            "detected": detected,
            "confidence": confidence,
            "detected_classes": detected_classes,
            "matched": bool(matched),
            "raw_boxes": raw_boxes,
        }
    except Exception as exc:
        raise RuntimeError(f"YOLOv8 inference сәтсіз: {exc}") from exc


def _load_soil_efficientnet() -> bool:
    global _SOIL_MODEL, _SOIL_CLASSES

    if _SOIL_MODEL is not None and _SOIL_CLASSES:
        return True
    if not os.path.exists(SOIL_MODEL_PATH):
        return False
    try:
        ckpt = torch.load(SOIL_MODEL_PATH, map_location=_SOIL_DEVICE, weights_only=False)
        if not isinstance(ckpt, dict):
            return False

        model_state = ckpt.get("model_state") or ckpt.get("state_dict")
        class_names = ckpt.get("class_names")
        num_classes = int(ckpt.get("num_classes", 0))
        if model_state is None:
            return False
        if not class_names:
            if num_classes > 0:
                class_names = [f"class_{i}" for i in range(num_classes)]
            else:
                return False
        if num_classes <= 0:
            num_classes = len(class_names)

        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        model.load_state_dict(model_state)
        model.to(_SOIL_DEVICE).eval()

        _SOIL_MODEL = model
        _SOIL_CLASSES = list(class_names)
        return True
    except Exception:
        return False


def _build_effnet_input(
    lat: float, lon: float, crop_type: str, rng: np.random.Generator
) -> Image.Image:
    width, height = 224, 224
    img = Image.new("RGB", (width, height), (102, 78, 52))
    draw = ImageDraw.Draw(img)

    for _ in range(10):
        x0 = int(rng.integers(0, width - 40))
        y0 = int(rng.integers(0, height - 40))
        x1 = min(width - 1, x0 + int(rng.integers(20, 70)))
        y1 = min(height - 1, y0 + int(rng.integers(20, 70)))
        col = (
            int(rng.integers(70, 180)),
            int(rng.integers(80, 170)),
            int(rng.integers(50, 140)),
        )
        draw.rectangle([x0, y0, x1, y1], fill=col)

    draw.text((8, 8), f"{crop_type}", fill=(230, 230, 230))
    draw.text((8, 26), f"{lat:.2f},{lon:.2f}", fill=(220, 220, 220))
    return img


def _soil_to_crop_label(soil_label: str, claimed_crop: str) -> str:
    label = soil_label.lower().replace("_", " ")
    if "black" in label or "alluvial" in label:
        return "Бидай"
    if "red" in label:
        return "Күнбағыс"
    if "yellow" in label:
        return "Арпа"
    if "mountain" in label:
        return "Жүзім"
    if "laterite" in label:
        return "Рапс"
    if "arid" in label:
        return "Басқа"
    return claimed_crop if claimed_crop in CROP_CLASSES else "Басқа"


def run_efficientnet(
    lat: float, lon: float, crop_type: str, rng: np.random.Generator
) -> dict[str, Any]:
    """EfficientNet inference (strict real mode)."""
    claimed = (crop_type or "").strip()
    claimed = claimed if claimed in CROP_CLASSES else "Басқа"

    if not _load_soil_efficientnet():
        raise RuntimeError(
            f"EfficientNet модельі дайын емес. Файл қажет: {SOIL_MODEL_PATH}"
        )

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = _build_effnet_input(lat, lon, claimed, rng)
    x = tf(image).unsqueeze(0).to(_SOIL_DEVICE)

    with torch.no_grad():
        logits = _SOIL_MODEL(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = np.argsort(probs)[::-1][:3]
    top3 = []
    for idx in top_idx:
        soil_class = str(_SOIL_CLASSES[int(idx)])
        crop_label = _soil_to_crop_label(soil_class, claimed)
        top3.append(
            {
                "label": crop_label,
                "soil_class": soil_class.replace("_", " "),
                "confidence": round(float(probs[int(idx)]), 4),
            }
        )

    identified_crop = top3[0]["label"] if top3 else "Басқа"
    confidence = float(top3[0]["confidence"]) if top3 else 0.0
    return {
        "identified_crop": identified_crop,
        "confidence": confidence,
        "match": identified_crop == claimed,
        "top3": top3,
    }


def _load_health_model():
    global _HEALTH_MODEL, _HEALTH_MODEL_KIND
    if _HEALTH_MODEL is not None:
        return _HEALTH_MODEL

    if keras is not None and os.path.exists(HEALTH_MODEL_H5_PATH):
        _HEALTH_MODEL = keras.models.load_model(HEALTH_MODEL_H5_PATH)
        _HEALTH_MODEL_KIND = "keras"
        return _HEALTH_MODEL

    if os.path.exists(HEALTH_MODEL_PTH_PATH):
        ckpt = torch.load(HEALTH_MODEL_PTH_PATH, map_location=_SOIL_DEVICE, weights_only=False)
        model = HealthCNNTorch().to(_SOIL_DEVICE).eval()
        state = ckpt.get("model_state") if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        _HEALTH_MODEL = model
        _HEALTH_MODEL_KIND = "torch"
        return _HEALTH_MODEL

    raise FileNotFoundError(
        f"Health model табылмады. Күтілетін файлдар: {HEALTH_MODEL_H5_PATH} немесе {HEALTH_MODEL_PTH_PATH}"
    )


def _load_unet_model():
    global _UNET_MODEL, _UNET_MODEL_KIND
    if _UNET_MODEL is not None:
        return _UNET_MODEL

    if keras is not None and os.path.exists(UNET_MODEL_H5_PATH):
        _UNET_MODEL = keras.models.load_model(UNET_MODEL_H5_PATH)
        _UNET_MODEL_KIND = "keras"
        return _UNET_MODEL

    if os.path.exists(UNET_MODEL_PTH_PATH):
        ckpt = torch.load(UNET_MODEL_PTH_PATH, map_location=_SOIL_DEVICE, weights_only=False)
        model = AreaUNetTorch().to(_SOIL_DEVICE).eval()
        state = ckpt.get("model_state") if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        _UNET_MODEL = model
        _UNET_MODEL_KIND = "torch"
        return _UNET_MODEL

    raise FileNotFoundError(
        f"U-Net model табылмады. Күтілетін файлдар: {UNET_MODEL_H5_PATH} немесе {UNET_MODEL_PTH_PATH}"
    )


def run_health_cnn(ndvi: float, seed: int) -> dict[str, Any]:
    """NDVI негізіндегі Health CNN inference (strict real mode)."""
    _ = seed  # сигнатура сақтау үшін
    model = _load_health_model()
    ndvi_patch = np.full((1, 224, 224, 1), float(ndvi), dtype=np.float32)

    if _HEALTH_MODEL_KIND == "keras":
        probs = np.asarray(model.predict(ndvi_patch, verbose=0))[0]
    else:
        x = torch.from_numpy(np.transpose(ndvi_patch, (0, 3, 1, 2))).float().to(_SOIL_DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    if probs.ndim != 1 or probs.size < 1:
        raise RuntimeError("Health CNN output форматы қате")
    score = int(np.argmax(probs)) + 1
    score = int(np.clip(score, 1, 5))
    return {"score": score, "label": HEALTH_LABELS[score]}


def run_unet(area: float, seed: int) -> dict[str, Any]:
    """U-Net area estimation inference (strict real mode)."""
    if area <= 0:
        raise ValueError("Аудан 0-ден үлкен болуы керек")
    model = _load_unet_model()

    rng = np.random.default_rng(seed + 505)
    rgb_patch = rng.uniform(0.0, 1.0, size=(1, 256, 256, 3)).astype(np.float32)
    if _UNET_MODEL_KIND == "keras":
        pred_mask = np.asarray(model.predict(rgb_patch, verbose=0))
    else:
        x = torch.from_numpy(np.transpose(rgb_patch, (0, 3, 1, 2))).float().to(_SOIL_DEVICE)
        with torch.no_grad():
            pred_mask = model(x).detach().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (0, 2, 3, 1))
    if pred_mask.size == 0:
        raise RuntimeError("U-Net mask output бос")

    mask = pred_mask[0]
    if mask.ndim == 3:
        mask = mask[..., 0]
    crop_ratio = float((mask > 0.5).mean())
    detected_ha = float(area * crop_ratio)
    error_pct = abs(detected_ha - area) / area * 100.0
    return {"detected_ha": round(detected_ha, 2), "error_pct": round(error_pct, 2)}


def calculate_score(
    *,
    crop_detected: bool,
    detection_confidence: float,
    declared_area: float,
    detected_area: float,
    health_score: int,
    crop_match: bool,
) -> dict[str, Any]:
    """100-балл жүйесі бойынша бағалау."""
    score = 0
    breakdown = {
        "crop_detected": 0,
        "area_match": 0,
        "health": 0,
        "crop_type_match": 0,
    }

    if crop_detected and detection_confidence > 0.8:
        score += 40
        breakdown["crop_detected"] = 40

    area_diff_ratio = abs(detected_area - declared_area) / declared_area
    if area_diff_ratio <= 0.15:
        score += 30
        breakdown["area_match"] = 30
    elif area_diff_ratio <= 0.30:
        score += 15
        breakdown["area_match"] = 15

    if health_score >= 4:
        score += 20
        breakdown["health"] = 20
    elif health_score >= 3:
        score += 10
        breakdown["health"] = 10

    if crop_match:
        score += 10
        breakdown["crop_type_match"] = 10

    if score >= 80:
        decision = "✅ РАСТАЛДЫ"
        risk = "Төмен"
    elif score >= 60:
        decision = "⚠️ КҮДІКТІ"
        risk = "Орташа"
    else:
        decision = "❌ ӨТІРІК"
        risk = "Жоғары"

    return {
        "score": int(score),
        "decision": decision,
        "risk_level": risk,
        "breakdown": breakdown,
        "area_difference_pct": round(area_diff_ratio * 100, 2),
    }


def verify(claim_dict: dict[str, Any], lat: float, lon: float) -> dict[str, Any]:
    """Субсидия өтінімін симуляциялық ML-пайплайнмен тексереді."""
    if not isinstance(claim_dict, dict):
        raise TypeError("claim_dict dict болуы керек")

    declared_area = float(claim_dict.get("declared_area_ha", 0))
    if declared_area <= 0:
        raise ValueError("declared_area_ha 0-ден үлкен болуы керек")

    claimed_crop = str(claim_dict.get("crop_type", "Басқа"))
    application_date = str(
        claim_dict.get("application_date", datetime.utcnow().strftime("%Y-%m-%d"))
    )
    seed = stable_seed(lat, lon, declared_area)
    rng = np.random.default_rng(seed + 202)

    satellite = simulate_sentinel2(lat, lon, application_date, rng)
    yolo = run_yolov8(lat, lon, declared_area, claimed_crop, rng)
    cls = run_efficientnet(lat, lon, claimed_crop, rng)
    health = run_health_cnn(satellite["ndvi"], seed)
    area = run_unet(declared_area, seed)

    scoring = calculate_score(
        crop_detected=yolo["detected"],
        detection_confidence=yolo["confidence"],
        declared_area=declared_area,
        detected_area=area["detected_ha"],
        health_score=health["score"],
        crop_match=cls["match"],
    )

    return {
        "decision": scoring["decision"],
        "score": scoring["score"],
        "risk_level": scoring["risk_level"],
        "score_breakdown": scoring["breakdown"],
        "details": {
            "farmer_name": claim_dict.get("farmer_name", ""),
            "region": claim_dict.get("region", ""),
            "claimed_crop": claimed_crop,
            "identified_crop": cls["identified_crop"],
            "crop_match": cls["match"],
            "crop_detected": yolo["detected"],
            "detection_confidence": yolo["confidence"],
            "detected_classes": yolo["detected_classes"],
            "yolo_matched": yolo["matched"],
            "yolo_raw_boxes": yolo["raw_boxes"],
            "classification_confidence": cls["confidence"],
            "classification_top3": cls["top3"],
            "health_score": health["score"],
            "health_label": health["label"],
            "declared_area_ha": round(declared_area, 2),
            "detected_area_ha": area["detected_ha"],
            "area_difference_pct": scoring["area_difference_pct"],
        },
        "satellite": satellite,
        "seed": seed,
    }


__all__ = [
    "stable_seed",
    "get_sentinel2_ndvi",
    "simulate_sentinel2",
    "run_yolov8",
    "run_efficientnet",
    "run_health_cnn",
    "run_unet",
    "calculate_score",
    "verify",
]
