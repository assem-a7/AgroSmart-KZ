"""
AgroSmart KZ — Forecasting Module
Прогнозирование заявок на субсидии по регионам и направлениям
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HW_AVAILABLE = True
except ImportError:
    _HW_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Подготовка временного ряда
# ─────────────────────────────────────────────────────────────
def prepare_ts(
    df: pd.DataFrame,
    region: str | None = None,
    direction: str | None = None,
) -> pd.DataFrame:
    """
    Агрегирует данные по месяцам.
    region / direction = None → все данные
    """
    data = df.copy()
    if region and region != "Барлығы":
        data = data[data["Область"] == region]
    if direction and direction != "Барлығы":
        data = data[data["Направление водства"] == direction]

    data["YM"] = data["Дата поступления"].dt.to_period("M")

    monthly = (
        data.groupby("YM")
        .agg(
            Заявок   =("Статус заявки", "count"),
            Сумма    =("Причитающая сумма", "sum"),
            Исполнено=("Статус заявки", lambda x: (x == "Исполнена").sum()),
        )
        .reset_index()
    )
    monthly["date"] = monthly["YM"].dt.to_timestamp()
    return monthly.sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Прогноз одного ряда
# ─────────────────────────────────────────────────────────────
def forecast_series(ts: pd.DataFrame, col: str = "Заявок", periods: int = 3) -> pd.DataFrame:
    """
    Holt-Winters ExponentialSmoothing (или fallback moving average).
    Возвращает объединённый DataFrame исторических + прогнозных данных.
    """
    values = ts[col].values.astype(float)

    if len(values) >= 4 and _HW_AVAILABLE:
        try:
            model  = ExponentialSmoothing(values, trend="add", seasonal=None,
                                          initialization_method="estimated")
            fit    = model.fit(optimized=True)
            fc_arr = np.maximum(fit.forecast(periods), 0)
        except Exception:
            fc_arr = _moving_avg_forecast(values, periods)
    else:
        fc_arr = _moving_avg_forecast(values, periods)

    last_date    = ts["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=periods, freq="MS"
    )

    history_df = ts[["date", col]].copy()
    history_df["is_forecast"] = False

    forecast_df = pd.DataFrame({
        "date":        future_dates,
        col:           fc_arr.round(0).astype(int),
        "is_forecast": True,
    })

    return pd.concat([history_df, forecast_df], ignore_index=True)


def _moving_avg_forecast(values: np.ndarray, periods: int) -> np.ndarray:
    """Simple trend-aware moving average fallback."""
    w   = min(3, len(values))
    avg = values[-w:].mean()
    trend = (values[-1] - values[-w]) / w if w > 1 else 0
    return np.array([max(0, avg + trend * (i + 1)) for i in range(periods)])


# ─────────────────────────────────────────────────────────────
# Сводная таблица прогнозов по всем регионам
# ─────────────────────────────────────────────────────────────
def regional_forecast(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    rows = []
    for region in sorted(df["Область"].unique()):
        ts = prepare_ts(df, region=region)
        if len(ts) < 2:
            continue
        fc     = forecast_series(ts, col="Заявок", periods=periods)
        future = fc[fc["is_forecast"]]

        rows.append({
            "Регион":            region,
            "Ағымдағы_ай":       int(ts["Заявок"].iloc[-1]),
            "Болжам_1_ай":       int(future["Заявок"].iloc[0]) if len(future) > 0 else 0,
            "Болжам_3_ай":       int(future["Заявок"].sum()),
            "Тренд":             "📈" if future["Заявок"].mean() > ts["Заявок"].mean() else "📉",
        })
    return pd.DataFrame(rows).sort_values("Болжам_3_ай", ascending=False)


# ─────────────────────────────────────────────────────────────
# Тренд по направлениям (для тепловой карты)
# ─────────────────────────────────────────────────────────────
def direction_trends(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["YM"]   = df2["Дата поступления"].dt.to_period("M")
    df2["date"] = df2["YM"].dt.to_timestamp()

    trend = (
        df2.groupby(["date", "Направление водства"])
        .agg(Заявок=("Статус заявки", "count"))
        .reset_index()
        .sort_values("date")
    )
    return trend
