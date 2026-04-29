"""
AgroSmart KZ — train_models.py
Запустить ОДИН РАЗ перед запуском приложения.
Читает Excel из data/, обучает модели, сохраняет результаты.

Использование:
    python train_models.py
"""

import os
import sys
import glob
import pandas as pd
from scoring_engine import ScoringEngine
from evaluate_models import generate_metrics_report


def find_xlsx(data_dir: str = "data") -> str | None:
    """Ищет первый xlsx-файл в папке data/"""
    files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    if not files:
        return None
    # Берём самый большой файл (основной датасет)
    return max(files, key=os.path.getsize)


def main():
    print("=" * 60)
    print("  AgroSmart KZ — Подготовка моделей")
    print("=" * 60)

    # 1. Найти файл данных
    xlsx_path = find_xlsx("data")
    if not xlsx_path:
        print("\n❌ Ошибка: xlsx-файл не найден в папке data/")
        print("   Положите файл выгрузки субсидий в папку data/ и повторите.")
        sys.exit(1)

    print(f"\n📂 Загрузка: {os.path.basename(xlsx_path)}")

    # 2. Загрузка (пропускаем первые 4 строки — заголовки госформы)
    try:
        df = pd.read_excel(xlsx_path, skiprows=4)
    except Exception:
        # Попытка без skiprows
        df = pd.read_excel(xlsx_path)

    print(f"   ✓ Загружено {len(df):,} строк, {len(df.columns)} столбцов")

    # 3. Запускаем пайплайн
    engine = ScoringEngine()
    df_processed = engine.run(df)

    # 4. Сохраняем модели
    engine.save("models/")

    # 5. Сохраняем обработанные данные
    os.makedirs("results", exist_ok=True)
    out_csv = "results/processed_data.csv"
    df_processed.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Данные сохранены → {out_csv}")

    # 6. Сохраняем региональный отчёт
    reg = engine.regional_report(df_processed)
    reg.to_csv("results/regional_report.csv", encoding="utf-8-sig")
    print(f"✅ Региональный отчёт → results/regional_report.csv")

    # 7. Шортлист топ-50
    sl = engine.shortlist(df_processed, top_n=50)
    sl.to_csv("results/shortlist.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Шортлист → results/shortlist.csv")

    # 8. Сохраняем метрики качества моделей
    metrics_path = generate_metrics_report(xlsx_path, output_dir="results")
    print(f"✅ Метрики моделей → {metrics_path}")

    print("\n" + "=" * 60)
    print("  Готово! Запустите приложение:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
