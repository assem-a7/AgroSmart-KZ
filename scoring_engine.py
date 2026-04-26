"""
AgroSmart KZ — Scoring Engine v2.0
Isolation Forest (аномалии) + XGBoost (Merit Score)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from config import DIRECTION_PRIORITY, KZ_REGIONS, ACTIVE_STATUSES, EXECUTED_STATUS, REJECTED_STATUS


class ScoringEngine:
    """
    ML-движок скоринга субсидий.
    Isolation Forest  → детекция аномалий / мошенничества
    XGBoost          → вероятность исполнения заявки
    Merit Score      → итоговый рейтинг заявителя (0–100)
    """

    def __init__(self):
        self.label_encoders: dict = {}
        self.scaler        = StandardScaler()
        self.iso_forest    = None
        self.xgb_model     = None
        self.feature_cols  : list = []
        self.df_processed  = None

    # ─────────────────────────────────────────────────────────
    # 1. ПРЕДОБРАБОТКА
    # ─────────────────────────────────────────────────────────
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Убираем пустые столбцы (Unnamed: ...)
        df = df.drop(
            columns=[c for c in df.columns if str(c).startswith("Unnamed")],
            errors="ignore",
        )

        # Обязательные столбцы
        required = ["Дата поступления", "Область", "Направление водства", "Статус заявки"]
        df = df.dropna(subset=[c for c in required if c in df.columns])

        # Парсинг даты
        df["Дата поступления"] = pd.to_datetime(
            df["Дата поступления"], format="%d.%m.%Y %H:%M:%S", errors="coerce"
        )
        df = df.dropna(subset=["Дата поступления"])

        # Признаки времени
        df["Месяц"]        = df["Дата поступления"].dt.month
        df["День_недели"]  = df["Дата поступления"].dt.dayofweek
        df["Час"]          = df["Дата поступления"].dt.hour
        df["Квартал"]      = df["Дата поступления"].dt.quarter
        df["Ночная_подача"] = ((df["Час"] >= 0) & (df["Час"] < 6)).astype(int)

        # Числовые
        df["Норматив"]          = pd.to_numeric(df["Норматив"],          errors="coerce").fillna(0)
        df["Причитающая сумма"] = pd.to_numeric(df["Причитающая сумма"], errors="coerce").fillna(0)

        # Кол-во голов = сумма / норматив
        df["Кол_голов"] = np.where(
            df["Норматив"] > 0,
            (df["Причитающая сумма"] / df["Норматив"]).round(0),
            0,
        )

        # Приоритет направления
        df["Направление_приоритет"] = (
            df["Направление водства"].map(DIRECTION_PRIORITY).fillna(0.5)
        )

        # Кодирование категорий
        for col in ["Область", "Направление водства", "Статус заявки"]:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        self.df_processed = df.copy()
        return df

    # ─────────────────────────────────────────────────────────
    # 2. ИСТОРИЯ ЗАЯВИТЕЛЯ
    # ─────────────────────────────────────────────────────────
    def build_history(self, df: pd.DataFrame):
        acol = "Акимат"         if "Акимат"         in df.columns else None
        rcol = "Район хозяйства" if "Район хозяйства" in df.columns else None

        if acol and rcol:
            df["applicant_id"] = df[acol].astype(str) + "|" + df[rcol].astype(str)
        elif acol:
            df["applicant_id"] = df[acol].astype(str)
        else:
            df["applicant_id"] = df["Область"].astype(str)

        history = (
            df.groupby("applicant_id")
            .agg(
                total_apps    =("Статус заявки", "count"),
                exec_count    =("Статус заявки", lambda x: (x == EXECUTED_STATUS).sum()),
                reject_count  =("Статус заявки", lambda x: (x == REJECTED_STATUS).sum()),
                avg_amount    =("Причитающая сумма", "mean"),
                avg_heads     =("Кол_голов", "mean"),
            )
            .reset_index()
        )
        history["success_rate"]   = (
            history["exec_count"] / history["total_apps"] * 100
        ).round(1)
        history["rejection_rate"] = (
            history["reject_count"] / history["total_apps"] * 100
        ).round(1)

        df = df.merge(history, on="applicant_id", how="left")
        return df, history

    # ─────────────────────────────────────────────────────────
    # 3. MERIT SCORE  (0–100, чем выше — тем лучше)
    # ─────────────────────────────────────────────────────────
    def calc_merit(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # A. История успеха (0–40)
        df["score_history"] = (df["success_rate"].fillna(0) / 100 * 40).round(2)

        # B. Масштаб хозяйства (0–25)
        max_heads = df["avg_heads"].quantile(0.95) or 1
        df["score_scale"] = (
            df["avg_heads"].clip(upper=max_heads) / max_heads * 25
        ).fillna(0).round(2)

        # C. Приоритет направления (0–20)
        df["score_direction"] = (df["Направление_приоритет"] * 20).round(2)

        # D. Опыт — кол-во заявок (0–15)
        max_apps = df["total_apps"].quantile(0.95) or 1
        df["score_experience"] = (
            df["total_apps"].clip(upper=max_apps) / max_apps * 15
        ).fillna(0).round(2)

        # E. Штраф за отказы (0 … -20)
        df["score_penalty"] = -(df["rejection_rate"].fillna(0) / 100 * 20).round(2)

        df["Merit_Score"] = (
            df["score_history"]
            + df["score_scale"]
            + df["score_direction"]
            + df["score_experience"]
            + df["score_penalty"]
        ).clip(0, 100).round(1)

        return df

    # ─────────────────────────────────────────────────────────
    # 4. ISOLATION FOREST — аномалии
    # ─────────────────────────────────────────────────────────
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Risk_Level"]   = "Normal"
        df["Risk_Score"]   = 0.0
        df["Risk_Reasons"] = ""

        feat_cols = [
            "Норматив", "Причитающая сумма", "Кол_голов",
            "Ночная_подача", "Месяц", "Направление_приоритет",
        ]
        X = df[feat_cols].fillna(0).values

        self.iso_forest = IsolationForest(
            n_estimators=300,
            contamination=0.03,   # ~3% считаем аномалиями
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        preds  = self.iso_forest.fit_predict(X)   # -1 = аномалия
        raw_sc = self.iso_forest.score_samples(X) # чем меньше — тем аномальнее

        # Нормируем: risk 0–100 (100 = максимальный риск)
        mn, mx = raw_sc.min(), raw_sc.max()
        risk = ((raw_sc - mx) / (mn - mx + 1e-9) * 100).clip(0, 100)
        df["Risk_Score"] = risk.round(1)

        # High Risk
        df.loc[preds == -1, "Risk_Level"] = "High Risk"
        # Medium Risk: граница (score > 60, но не -1)
        med_mask = (preds == 1) & (risk > 60)
        df.loc[med_mask, "Risk_Level"] = "Medium Risk"

        # Дополнительные правила-объяснения
        zero_mask = (df["Причитающая сумма"] == 0) & (df["Норматив"] > 0)
        df.loc[zero_mask, "Risk_Level"]   = "Medium Risk"
        df.loc[zero_mask, "Risk_Reasons"] = "Норматив бар, бірақ сумма нөл"

        night_mask = df["Ночная_подача"] == 1
        df.loc[night_mask & (df["Risk_Level"] == "High Risk"), "Risk_Reasons"] = (
            "Түнгі уақытта берілген + статистикалық аномалия"
        )

        return df

    # ─────────────────────────────────────────────────────────
    # 5. XGBOOST — обучение на исторических данных
    # ─────────────────────────────────────────────────────────
    def train_xgboost(self, df: pd.DataFrame) -> pd.DataFrame:
        historical = df[
            df["Статус заявки"].isin([EXECUTED_STATUS, REJECTED_STATUS])
        ].copy()

        if len(historical) < 50:
            df["XGB_Prob"]    = 0.5
            df["Final_Score"] = df["Merit_Score"]
            return df

        historical["target"] = (historical["Статус заявки"] == EXECUTED_STATUS).astype(int)

        candidate_features = [
            "Норматив", "Причитающая сумма", "Кол_голов",
            "Месяц", "День_недели", "Час", "Квартал", "Ночная_подача",
            "Направление_приоритет",
            "success_rate", "rejection_rate", "total_apps",
            "avg_amount", "avg_heads",
            "Область_enc", "Направление водства_enc",
        ]
        self.feature_cols = [c for c in candidate_features if c in historical.columns]

        X = historical[self.feature_cols].fillna(0)
        y = historical["target"]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        X_all = df[self.feature_cols].fillna(0)
        df["XGB_Prob"] = self.xgb_model.predict_proba(X_all)[:, 1].round(3)

        # Итоговый скор: 60% XGBoost + 40% Merit
        df["Final_Score"] = (
            df["XGB_Prob"] * 60 + df["Merit_Score"] * 0.4
        ).clip(0, 100).round(1)

        return df

    # ─────────────────────────────────────────────────────────
    # 6. SHORTLIST
    # ─────────────────────────────────────────────────────────
    def shortlist(
        self,
        df: pd.DataFrame,
        top_n: int = 20,
        direction: str = None,
        region: str = None,
    ) -> pd.DataFrame:
        active = df[df["Статус заявки"].isin(ACTIVE_STATUSES)].copy()
        if direction:
            active = active[active["Направление водства"] == direction]
        if region:
            active = active[active["Область"] == region]

        sort_col = "Final_Score" if "Final_Score" in active.columns else "Merit_Score"
        active = active.sort_values(sort_col, ascending=False)

        show = [
            "Номер заявки", "Дата поступления", "Область",
            "Направление водства", "Норматив", "Причитающая сумма",
            "Кол_голов", "Merit_Score", "Final_Score", "XGB_Prob",
            "success_rate", "total_apps", "Risk_Level", "Risk_Score",
        ]
        show = [c for c in show if c in active.columns]
        return active[show].head(top_n).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────
    # 7. РЕГИОНАЛЬНЫЙ ОТЧЁТ
    # ─────────────────────────────────────────────────────────
    def regional_report(self, df: pd.DataFrame) -> pd.DataFrame:
        rep = (
            df.groupby("Область")
            .agg(
                Всего_заявок   =("Статус заявки", "count"),
                Исполнено      =("Статус заявки", lambda x: (x == EXECUTED_STATUS).sum()),
                Отклонено      =("Статус заявки", lambda x: (x == REJECTED_STATUS).sum()),
                Общая_сумма    =("Причитающая сумма", "sum"),
                Средняя_сумма  =("Причитающая сумма", "mean"),
                Средний_Merit  =("Merit_Score", "mean"),
                High_Risk      =("Risk_Level", lambda x: (x == "High Risk").sum()),
            )
            .round(1)
        )
        rep["Процент_исполн"] = (rep["Исполнено"] / rep["Всего_заявок"] * 100).round(1)
        rep["Процент_риска"]  = (rep["High_Risk"] / rep["Всего_заявок"] * 100).round(1)

        # Добавляем координаты для карты
        for region, info in KZ_REGIONS.items():
            if region in rep.index:
                rep.loc[region, "lat"] = info["lat"]
                rep.loc[region, "lon"] = info["lon"]

        return rep.sort_values("Средний_Merit", ascending=False)

    # ─────────────────────────────────────────────────────────
    # 8. ПОЛНЫЙ ПАЙПЛАЙН
    # ─────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        print("=" * 60)
        print("AgroSmart KZ — Запуск пайплайна")
        print("=" * 60)

        print("\n🔧 [1/5] Предобработка данных...")
        df = self.preprocess(df)
        print(f"   ✓ {len(df):,} записей обработано")

        print("\n📊 [2/5] История заявителей...")
        df, history = self.build_history(df)
        print(f"   ✓ {len(history):,} уникальных заявителей")

        print("\n🎯 [3/5] Merit Score...")
        df = self.calc_merit(df)
        print(f"   ✓ Ср. Merit Score: {df['Merit_Score'].mean():.1f}")

        print("\n🚨 [4/5] Isolation Forest — аномалии...")
        df = self.detect_anomalies(df)
        rc = df["Risk_Level"].value_counts()
        print(f"   ✓ Normal:      {rc.get('Normal', 0):,}")
        print(f"   ✓ Medium Risk: {rc.get('Medium Risk', 0):,}")
        print(f"   ✓ High Risk:   {rc.get('High Risk', 0):,}")

        print("\n🤖 [5/5] XGBoost обучение...")
        df = self.train_xgboost(df)
        if "XGB_Prob" in df.columns:
            print(f"   ✓ Ср. вер. исполнения: {df['XGB_Prob'].mean():.3f}")

        print("\n✅ Пайплайн завершён!\n")
        self.df_processed = df
        return df

    # ─────────────────────────────────────────────────────────
    # Сохранение / загрузка моделей
    # ─────────────────────────────────────────────────────────
    def save(self, path: str = "models/") -> None:
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.iso_forest,     f"{path}iso_forest.pkl")
        joblib.dump(self.xgb_model,      f"{path}xgb_model.pkl")
        joblib.dump(self.label_encoders, f"{path}label_encoders.pkl")
        joblib.dump(self.feature_cols,   f"{path}feature_cols.pkl")
        print(f"✅ Модели сохранены → {path}")

    def load(self, path: str = "models/") -> bool:
        try:
            self.iso_forest     = joblib.load(f"{path}iso_forest.pkl")
            self.xgb_model      = joblib.load(f"{path}xgb_model.pkl")
            self.label_encoders = joblib.load(f"{path}label_encoders.pkl")
            self.feature_cols   = joblib.load(f"{path}feature_cols.pkl")
            return True
        except Exception as e:
            print(f"⚠ Модели не загружены: {e}")
            return False
