"""Model evaluation helpers for AgroSmart KZ.

Builds an honest time-based holdout evaluation and saves model metrics to JSON.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from config import ACTIVE_STATUSES, DIRECTION_PRIORITY, EXECUTED_STATUS, REJECTED_STATUS
from forecasting import forecast_series, prepare_ts


def find_xlsx(data_dir: str = "data") -> str | None:
    files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    if not files:
        return None
    return max(files, key=os.path.getsize)


def _load_raw_dataframe(xlsx_path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(xlsx_path, skiprows=4)
    except Exception:
        return pd.read_excel(xlsx_path)


def _applicant_id_frame(df: pd.DataFrame) -> pd.Series:
    acol = "Акимат" if "Акимат" in df.columns else None
    rcol = "Район хозяйства" if "Район хозяйства" in df.columns else None

    if acol and rcol:
        return df[acol].astype(str) + "|" + df[rcol].astype(str)
    if acol:
        return df[acol].astype(str)
    return df["Область"].astype(str)


def _build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.drop(columns=[c for c in frame.columns if str(c).startswith("Unnamed")], errors="ignore")
    frame["Дата поступления"] = pd.to_datetime(frame["Дата поступления"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    frame = frame.dropna(subset=["Дата поступления", "Область", "Направление водства", "Статус заявки"])

    frame["Норматив"] = pd.to_numeric(frame.get("Норматив"), errors="coerce").fillna(0)
    frame["Причитающая сумма"] = pd.to_numeric(frame.get("Причитающая сумма"), errors="coerce").fillna(0)
    frame["Месяц"] = frame["Дата поступления"].dt.month
    frame["День_недели"] = frame["Дата поступления"].dt.dayofweek
    frame["Час"] = frame["Дата поступления"].dt.hour
    frame["Квартал"] = frame["Дата поступления"].dt.quarter
    frame["Ночная_подача"] = ((frame["Час"] >= 0) & (frame["Час"] < 6)).astype(int)
    frame["Кол_голов"] = np.where(frame["Норматив"] > 0, (frame["Причитающая сумма"] / frame["Норматив"]).round(0), 0)
    frame["Направление_приоритет"] = frame["Направление водства"].map(DIRECTION_PRIORITY).fillna(0.5)
    frame["applicant_id"] = _applicant_id_frame(frame)
    return frame


def _build_temporal_history(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("Дата поступления").copy()
    ordered["total_apps"] = 0.0
    ordered["exec_count"] = 0.0
    ordered["reject_count"] = 0.0
    ordered["avg_amount"] = 0.0
    ordered["avg_heads"] = 0.0

    for _, group in ordered.groupby("applicant_id", sort=False):
        idx = group.index
        total_apps = np.arange(len(group), dtype=float)
        exec_count = group["Статус заявки"].eq(EXECUTED_STATUS).cumsum().shift(1).fillna(0).astype(float)
        reject_count = group["Статус заявки"].eq(REJECTED_STATUS).cumsum().shift(1).fillna(0).astype(float)
        avg_amount = group["Причитающая сумма"].expanding().mean().shift(1).fillna(0.0)
        avg_heads = group["Кол_голов"].expanding().mean().shift(1).fillna(0.0)

        ordered.loc[idx, "total_apps"] = total_apps
        ordered.loc[idx, "exec_count"] = exec_count.values
        ordered.loc[idx, "reject_count"] = reject_count.values
        ordered.loc[idx, "avg_amount"] = avg_amount.values
        ordered.loc[idx, "avg_heads"] = avg_heads.values

    ordered["success_rate"] = np.where(ordered["total_apps"] > 0, ordered["exec_count"] / ordered["total_apps"] * 100, 0.0)
    ordered["rejection_rate"] = np.where(ordered["total_apps"] > 0, ordered["reject_count"] / ordered["total_apps"] * 100, 0.0)
    ordered["total_apps"] = ordered["total_apps"].astype(int)
    return ordered


def _time_split(frame: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
    ordered = frame.sort_values("Дата поступления").reset_index(drop=True)
    if len(ordered) < 10:
        return ordered, ordered.iloc[0:0].copy(), None

    split_idx = max(1, int(len(ordered) * (1.0 - test_ratio)))
    split_idx = min(split_idx, len(ordered) - 1)
    split_date = ordered.loc[split_idx, "Дата поступления"]
    train = ordered.iloc[:split_idx].copy()
    test = ordered.iloc[split_idx:].copy()
    return train, test, split_date


def _encode_categories(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    for col in ["Область", "Направление водства", "Статус заявки"]:
        encoder = LabelEncoder()
        encoder.fit(train[col].astype(str))

        train[f"{col}_enc"] = encoder.transform(train[col].astype(str))
        lookup = {value: idx for idx, value in enumerate(encoder.classes_)}
        test[f"{col}_enc"] = test[col].astype(str).map(lookup).fillna(-1).astype(int)

    return train, test


def _evaluate_classification(frame: pd.DataFrame) -> dict[str, Any]:
    evaluated = _build_temporal_history(frame)
    train, test, split_date = _time_split(evaluated)
    if len(test) == 0:
        return {"status": "skipped", "reason": "not enough rows for time split"}

    train, test = _encode_categories(train, test)
    train = train[train["Статус заявки"].isin([EXECUTED_STATUS, REJECTED_STATUS])].copy()
    test = test[test["Статус заявки"].isin([EXECUTED_STATUS, REJECTED_STATUS])].copy()

    if len(train) < 20 or len(test) < 10:
        return {"status": "skipped", "reason": "insufficient labeled data for evaluation"}

    train["target"] = (train["Статус заявки"] == EXECUTED_STATUS).astype(int)
    test["target"] = (test["Статус заявки"] == EXECUTED_STATUS).astype(int)

    feature_candidates = [
        "Норматив",
        "Причитающая сумма",
        "Кол_голов",
        "Месяц",
        "День_недели",
        "Час",
        "Квартал",
        "Ночная_подача",
        "Направление_приоритет",
        "success_rate",
        "rejection_rate",
        "total_apps",
        "avg_amount",
        "avg_heads",
        "Область_enc",
        "Направление водства_enc",
    ]
    feature_cols = [col for col in feature_candidates if col in train.columns and col in test.columns]

    x_train = train[feature_cols].fillna(0)
    y_train = train["target"]
    x_test = test[feature_cols].fillna(0)
    y_test = test["target"]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    probs = model.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, probs)) if y_test.nunique() > 1 else None
    pr_auc = float(average_precision_score(y_test, probs)) if y_test.nunique() > 1 else None
    conf = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
    precision = float(precision_score(y_test, preds, zero_division=0))
    recall = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    brier = float(brier_score_loss(y_test, probs))

    if len(np.unique(probs)) > 1 and len(y_test) >= 10:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=min(10, max(2, len(y_test) // 10)), strategy="uniform")
        calibration_bins = [
            {"mean_predicted_prob": float(mp), "fraction_of_positives": float(fp)}
            for mp, fp in zip(mean_pred, frac_pos)
        ]
    else:
        calibration_bins = []

    return {
        "status": "ok",
        "split_date": split_date.isoformat() if split_date is not None else None,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": feature_cols,
        "metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "brier_score": brier,
            "confusion_matrix": conf,
            "classification_report": report,
            "calibration_curve": calibration_bins,
        },
    }


def _evaluate_forecasting(frame: pd.DataFrame) -> dict[str, Any]:
    ts = prepare_ts(frame)
    if len(ts) < 4:
        return {"status": "skipped", "reason": "not enough monthly history"}

    horizon = min(6, max(1, len(ts) // 4))
    train_ts = ts.iloc[:-horizon].copy()
    test_ts = ts.iloc[-horizon:].copy()
    if len(train_ts) < 2:
        return {"status": "skipped", "reason": "not enough history for forecast evaluation"}

    results: dict[str, Any] = {"status": "ok", "horizon": int(horizon)}

    for column in ["Заявок", "Сумма"]:
        forecast_frame = forecast_series(train_ts[["date", column]].copy(), col=column, periods=horizon)
        forecast_only = forecast_frame[forecast_frame["is_forecast"]][column].astype(float).values
        actual = test_ts[column].astype(float).values[: len(forecast_only)]

        if len(actual) == 0:
            continue

        mae = float(mean_absolute_error(actual, forecast_only))
        denom = np.where(actual == 0, np.nan, actual)
        mape = float(np.nanmean(np.abs((actual - forecast_only) / denom)) * 100)

        results[column] = {
            "mae": mae,
            "mape": None if np.isnan(mape) else mape,
            "actual": [float(v) for v in actual.tolist()],
            "forecast": [float(v) for v in forecast_only.tolist()],
        }

    return results


def _evaluate_anomalies(frame: pd.DataFrame) -> dict[str, Any]:
    evaluation_frame = _build_temporal_history(frame)
    feature_cols = [
        "Норматив",
        "Причитающая сумма",
        "Кол_голов",
        "Ночная_подача",
        "Месяц",
        "Направление_приоритет",
    ]
    x_values = evaluation_frame[feature_cols].fillna(0).values

    model = IsolationForest(
        n_estimators=300,
        contamination=0.03,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_values)
    preds = model.predict(x_values)
    high_risk_share = float((preds == -1).mean() * 100)

    return {
        "status": "ok",
        "contamination": 0.03,
        "high_risk_share_percent": high_risk_share,
    }


def generate_metrics_report(xlsx_path: str, output_dir: str = "results") -> str:
    raw = _load_raw_dataframe(xlsx_path)
    frame = _build_base_features(raw)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_file": os.path.basename(xlsx_path),
        "records": int(len(frame)),
        "active_rows": int(frame[frame["Статус заявки"].isin(ACTIVE_STATUSES)].shape[0]),
        "classification": _evaluate_classification(frame),
        "forecasting": _evaluate_forecasting(frame),
        "anomaly_detection": _evaluate_anomalies(frame),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    return out_path


if __name__ == "__main__":
    xlsx = find_xlsx("data")
    if not xlsx:
        raise SystemExit("No xlsx file found in data/")
    path = generate_metrics_report(xlsx)
    print(f"Saved metrics to {path}")