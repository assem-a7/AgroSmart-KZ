"""
AgroSmart KZ — Негізгі қосымша
Іске қосу: streamlit run app.py
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from config import KZ_REGIONS, REGION_LIST, DIRECTION_PRIORITY
from scoring_engine import ScoringEngine
from soil_analyzer import analyze_photos, SOIL_INFO
from forecasting import prepare_ts, forecast_series, regional_forecast, direction_trends

# ─────────────────────────────────────────────────────────────
# БЕТ КОНФИГУРАЦИЯСЫ
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroSmart KZ",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS — КӘСІБИ ДИЗАЙН
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
html, body { background: #ffffff !important; }
.stApp { background: #ffffff !important; }
.main .block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1280px; }

section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04);
}
section[data-testid="stSidebar"] * { color: #111827 !important; }
section[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
section[data-testid="stSidebar"] .stRadio label {
    border-radius: 8px !important;
    padding: 10px 12px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #374151 !important;
    transition: all 0.15s ease;
    cursor: pointer;
    border: 1px solid transparent !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #f0fdf4 !important;
    color: #166534 !important;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 22px 24px 18px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s;
}
div[data-testid="stMetric"]:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
div[data-testid="stMetric"]::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #16a34a, #4ade80);
    border-radius: 0 0 14px 14px;
}
div[data-testid="stMetric"] label {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #9ca3af !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
div[data-testid="stMetric"] > div > div {
    font-size: 28px !important;
    font-weight: 800 !important;
    color: #111827 !important;
    letter-spacing: -0.5px;
}

div.stButton > button {
    background: #16a34a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 11px 26px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(22,163,74,0.25) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background: #15803d !important;
    box-shadow: 0 4px 16px rgba(22,163,74,0.35) !important;
    transform: translateY(-1px) !important;
}

.agro-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}
.agro-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.agro-card-green  { border-top: 3px solid #16a34a; }
.agro-card-red    { border-top: 3px solid #dc2626; }
.agro-card-orange { border-top: 3px solid #d97706; }
.agro-card-blue   { border-top: 3px solid #2563eb; }

.page-header {
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid #f3f4f6;
}
.page-title {
    font-size: 26px;
    font-weight: 800;
    color: #111827;
    margin: 0 0 4px;
    letter-spacing: -0.5px;
}
.page-subtitle {
    font-size: 13px;
    color: #9ca3af;
    margin: 0;
    font-weight: 400;
}

.section-title {
    font-size: 14px;
    font-weight: 700;
    color: #374151;
    margin: 28px 0 12px;
    letter-spacing: 0.1px;
}

.info-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #166534;
    font-weight: 500;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

hr { border-color: #f3f4f6 !important; margin: 20px 0 !important; }

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ДЕРЕКТЕРДІ ЖҮКТЕУ
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Деректер жүктелуде...")
def load_data() -> pd.DataFrame | None:
    csv_path = "results/processed_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        if "Дата поступления" in df.columns:
            df["Дата поступления"] = pd.to_datetime(df["Дата поступления"], errors="coerce")
        return df

    xlsx_files = glob.glob("data/*.xlsx")
    if not xlsx_files:
        return None

    xlsx_path = max(xlsx_files, key=os.path.getsize)
    try:
        raw = pd.read_excel(xlsx_path, skiprows=4)
    except Exception:
        raw = pd.read_excel(xlsx_path)

    engine = ScoringEngine()
    df = engine.run(raw)
    os.makedirs("results", exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    engine.save("models/")
    return df


# ─────────────────────────────────────────────────────────────
# БҮЙІР ПАНЕЛІ
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 20px; border-bottom:1px solid #f3f4f6; margin-bottom:12px;">
        <div style="font-size:20px; font-weight:800; color:#111827; letter-spacing:-0.5px;">
            🌾 AgroSmart KZ
        </div>
        <div style="font-size:11px; color:#9ca3af; margin-top:3px; font-weight:400;">
            Жасанды интеллект · Ауылшаруашылық
        </div>
        <div style="display:inline-block; background:#dcfce7; color:#166534; font-size:10px;
             font-weight:700; padding:2px 8px; border-radius:20px; margin-top:6px;
             letter-spacing:0.5px;">AI ПЛАТФОРМАСЫ</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Навигация",
        [
            "Жалпы шолу",
            "Аймақтық карта",
            "AI Скоринг",
            "Болжамдау",
            "Жер талдауы",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("""
    <div style="font-size:11px; color:#64748b; line-height:1.8;">
        <div>📁 Дерек көзі: ҚР субсидия тізімі 2025</div>
        <div>📍 17 облыс + 3 қала</div>
        <div>🤖 Isolation Forest + XGBoost</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ДЕРЕКТЕРДІ ЖҮКТЕУ
# ─────────────────────────────────────────────────────────────
df = load_data()

if df is None:
    st.markdown("""
    <div class="agro-card agro-card-red">
        <h3 style="color:#dc2626; margin:0 0 8px;">⚠️ Деректер табылмады</h3>
        <p style="color:#6b7280; margin:0;">
            1. <code>data/</code> папкасына xlsx файлын қойыңыз<br>
            2. <code>python train_models.py</code> іске қосыңыз<br>
            3. <code>streamlit run app.py</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Жалпы көрсеткіштер
total_apps  = len(df)
total_sum   = df["Причитающая сумма"].sum() if "Причитающая сумма" in df.columns else 0
high_risk_n = int((df["Risk_Level"] == "High Risk").sum()) if "Risk_Level" in df.columns else 0
exec_pct    = round((df["Статус заявки"] == "Исполнена").sum() / total_apps * 100, 1) if total_apps else 0
avg_merit   = round(df["Merit_Score"].mean(), 1) if "Merit_Score" in df.columns else 0


def make_chart_layout(height=340):
    return dict(
        height=height,
        margin=dict(t=36, b=10, l=10, r=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter", size=12, color="#374151"),
    )


# ═══════════════════════════════════════════════════════════════
# БЕТ 1: ЖАЛПЫ ШОЛУ
# ═══════════════════════════════════════════════════════════════
if page == "Жалпы шолу":
    st.markdown("""
    <div class="page-header">
        <p class="page-title">Жалпы шолу</p>
        <p class="page-subtitle">Субсидия өтінімдерінің жиынтық аналитикасы · 2025 жыл</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Барлық өтінім", f"{total_apps:,}")
    c2.metric("Жалпы сомасы", f"{total_sum/1e9:.2f} млрд ₸")
    c3.metric("Орындалу үлесі", f"{exec_pct}%")
    c4.metric("Жоғары тәуекел", f"{high_risk_n:,}")
    c5.metric("Орташа Merit", f"{avg_merit}/100")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="section-title">Өтінімдер статусы бойынша</p>', unsafe_allow_html=True)
        status_counts = df["Статус заявки"].value_counts().reset_index()
        status_counts.columns = ["Статус", "Саны"]
        fig = px.pie(
            status_counts, names="Статус", values="Саны",
            color_discrete_sequence=["#1a6b3c","#2ecc71","#a7f3d0","#d97706","#dc2626"],
            hole=0.5,
        )
        fig.update_layout(**make_chart_layout())
        fig.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-title">Субсидия бағыттары (ТОП-10)</p>', unsafe_allow_html=True)
        if "Направление водства" in df.columns:
            dir_cnt = df["Направление водства"].value_counts().head(10).reset_index()
            dir_cnt.columns = ["Бағыт", "Саны"]
            fig2 = px.bar(
                dir_cnt, x="Саны", y="Бағыт", orientation="h",
                color="Саны",
                color_continuous_scale=["#a7f3d0","#1a6b3c"],
            )
            fig2.update_layout(**make_chart_layout(), coloraxis_showscale=False)
            fig2.update_layout(yaxis_title="", xaxis_title="Өтінімдер саны")
            st.plotly_chart(fig2, use_container_width=True)

    # Айлық динамика
    if "Дата поступления" in df.columns:
        st.markdown('<p class="section-title">Өтінімдер динамикасы (ай бойынша)</p>', unsafe_allow_html=True)
        ts_all = prepare_ts(df)
        if len(ts_all) > 0:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=ts_all["date"], y=ts_all["Заявок"],
                fill="tozeroy",
                line=dict(color="#1a6b3c", width=2.5),
                fillcolor="rgba(26,107,60,0.1)",
                name="Өтінімдер саны",
            ))
            fig3.update_layout(**make_chart_layout(260))
            fig3.update_layout(xaxis_title="Ай", yaxis_title="Өтінімдер саны", showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    # Тәуекел
    st.markdown('<p class="section-title">Тәуекел деңгейі бойынша бөліну</p>', unsafe_allow_html=True)
    if "Risk_Level" in df.columns:
        rc = df["Risk_Level"].value_counts().reset_index()
        rc.columns = ["Деңгей", "Саны"]
        cmap = {"Normal": "#1a6b3c", "Medium Risk": "#d97706", "High Risk": "#dc2626"}
        fig4 = px.bar(
            rc, x="Деңгей", y="Саны",
            color="Деңгей", color_discrete_map=cmap,
            text="Саны",
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(**make_chart_layout(260), showlegend=False)
        fig4.update_layout(xaxis_title="", yaxis_title="Өтінімдер саны")
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# БЕТ 2: АЙМАҚТЫҚ КАРТА
# ═══════════════════════════════════════════════════════════════
elif page == "Аймақтық карта":
    st.markdown("""
    <div class="page-header">
        <p class="page-title">Аймақтық карта</p>
        <p class="page-subtitle">Қазақстан облыстары бойынша субсидия аналитикасы</p>
    </div>
    """, unsafe_allow_html=True)

    engine = ScoringEngine()
    reg_df = engine.regional_report(df)
    reg_reset = reg_df.copy().reset_index()

    # Координаттар қосу
    if "lat" not in reg_reset.columns:
        reg_reset["lat"] = reg_reset["Область"].map(
            lambda r: KZ_REGIONS.get(r, {}).get("lat", 51.0))
        reg_reset["lon"] = reg_reset["Область"].map(
            lambda r: KZ_REGIONS.get(r, {}).get("lon", 71.0))

    reg_reset["lat"] = pd.to_numeric(reg_reset["lat"], errors="coerce").fillna(51.0)
    reg_reset["lon"] = pd.to_numeric(reg_reset["lon"], errors="coerce").fillna(71.0)

    fig_map = px.scatter_mapbox(
        reg_reset,
        lat="lat", lon="lon",
        size="Всего_заявок",
        color="Средний_Merit",
        hover_name="Область",
        hover_data={
            "Всего_заявок": True,
            "Процент_исполн": True,
            "Средний_Merit": ":.1f",
            "Процент_риска": True,
            "lat": False, "lon": False,
        },
        color_continuous_scale=["#dc2626","#d97706","#1a6b3c"],
        size_max=55,
        zoom=3.8,
        center={"lat": 48.5, "lon": 67.0},
        mapbox_style="open-street-map",
    )
    fig_map.update_layout(
        height=500,
        margin=dict(r=0, t=0, l=0, b=0),
        coloraxis_colorbar=dict(title="Merit"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        💡 Шеңбер өлшемі — өтінімдер санына сәйкес. Түсі — Merit Score деңгейін көрсетеді (жасыл = жоғары, қызыл = төмен).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Облыстар бойынша толық кесте</p>', unsafe_allow_html=True)

    display_cols = {
        "Всего_заявок":   "Барлық өтінім",
        "Исполнено":      "Орындалды",
        "Процент_исполн": "Орындалу %",
        "Средний_Merit":  "Орташа Merit",
        "High_Risk":      "Жоғары тәуекел",
        "Процент_риска":  "Тәуекел %",
        "Общая_сумма":    "Жалпы сомасы (₸)",
    }
    show_df = reg_df[[c for c in display_cols if c in reg_df.columns]].copy()
    show_df = show_df.rename(columns=display_cols)
    if "Жалпы сомасы (₸)" in show_df.columns:
        show_df["Жалпы сомасы (₸)"] = show_df["Жалпы сомасы (₸)"].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
    st.dataframe(show_df, use_container_width=True, height=520)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Merit Score рейтингі (ТОП-10)</p>', unsafe_allow_html=True)
        top_m = reg_df["Средний_Merit"].dropna().sort_values(ascending=False).head(10)
        fig_m = px.bar(x=top_m.values, y=top_m.index, orientation="h",
                       color=top_m.values,
                       color_continuous_scale=["#d97706","#1a6b3c"])
        fig_m.update_layout(**make_chart_layout(360), coloraxis_showscale=False)
        fig_m.update_layout(yaxis_title="", xaxis_title="Merit Score")
        st.plotly_chart(fig_m, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Жоғары тәуекел үлесі (ТОП-10)</p>', unsafe_allow_html=True)
        top_r = reg_df["Процент_риска"].dropna().sort_values(ascending=False).head(10)
        fig_r = px.bar(x=top_r.values, y=top_r.index, orientation="h",
                       color=top_r.values,
                       color_continuous_scale=["#1a6b3c","#dc2626"])
        fig_r.update_layout(**make_chart_layout(360), coloraxis_showscale=False)
        fig_r.update_layout(yaxis_title="", xaxis_title="Тәуекел %")
        st.plotly_chart(fig_r, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# БЕТ 3: AI СКОРИНГ
# ═══════════════════════════════════════════════════════════════
elif page == "AI Скоринг":
    st.markdown("""
    <div class="page-header">
        <p class="page-title">AI Скоринг жүйесі</p>
        <p class="page-subtitle">Isolation Forest · XGBoost · Merit Score алгоритмдері</p>
    </div>
    """, unsafe_allow_html=True)

    # Сүзгілер
    st.markdown('<p class="section-title">Сүзгілер</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)

    with f1:
        regions_avail = ["Барлығы"] + sorted(df["Область"].unique().tolist())
        sel_region = st.selectbox("Облыс", regions_avail)
    with f2:
        dirs_avail = (["Барлығы"] + sorted(df["Направление водства"].unique().tolist())
                      if "Направление водства" in df.columns else ["Барлығы"])
        sel_dir = st.selectbox("Субсидия бағыты", dirs_avail)
    with f3:
        sel_risk = st.selectbox("Тәуекел деңгейі",
                                ["Барлығы", "Қалыпты", "Орташа тәуекел", "Жоғары тәуекел"])

    risk_map = {"Барлығы": None, "Қалыпты": "Normal",
                "Орташа тәуекел": "Medium Risk", "Жоғары тәуекел": "High Risk"}

    filtered = df.copy()
    if sel_region != "Барлығы":
        filtered = filtered[filtered["Область"] == sel_region]
    if sel_dir != "Барлығы" and "Направление водства" in filtered.columns:
        filtered = filtered[filtered["Направление водства"] == sel_dir]
    if risk_map[sel_risk] and "Risk_Level" in filtered.columns:
        filtered = filtered[filtered["Risk_Level"] == risk_map[sel_risk]]

    st.markdown(f"""
    <div class="info-box">
        Сүзгі нәтижесі: <strong>{len(filtered):,}</strong> өтінім табылды
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Барлық өтінім", f"{len(filtered):,}")
    k2.metric("Жоғары тәуекел",
              f"{(filtered['Risk_Level'] == 'High Risk').sum():,}"
              if "Risk_Level" in filtered.columns else "—")
    k3.metric("Орташа Merit",
              f"{filtered['Merit_Score'].mean():.1f}"
              if "Merit_Score" in filtered.columns else "—")
    k4.metric("Орындалу ықтималы",
              f"{filtered['XGB_Prob'].mean():.1%}"
              if "XGB_Prob" in filtered.columns else "—")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-title">Merit Score таралуы</p>', unsafe_allow_html=True)
        if "Merit_Score" in filtered.columns:
            fig_h = px.histogram(filtered, x="Merit_Score", nbins=40,
                                 color_discrete_sequence=["#1a6b3c"])
            fig_h.update_layout(**make_chart_layout(280))
            fig_h.update_layout(xaxis_title="Merit Score", yaxis_title="Өтінімдер саны")
            st.plotly_chart(fig_h, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-title">Тәуекел деңгейлері</p>', unsafe_allow_html=True)
        if "Risk_Level" in filtered.columns:
            rc = filtered["Risk_Level"].value_counts().reset_index()
            rc.columns = ["Деңгей", "Саны"]
            rc["Деңгей"] = rc["Деңгей"].map({"Normal": "Қалыпты",
                                              "Medium Risk": "Орташа",
                                              "High Risk": "Жоғары"})
            cmap = {"Қалыпты": "#1a6b3c", "Орташа": "#d97706", "Жоғары": "#dc2626"}
            fig_rc = px.pie(rc, names="Деңгей", values="Саны",
                            color="Деңгей", color_discrete_map=cmap, hole=0.5)
            fig_rc.update_layout(**make_chart_layout(280))
            st.plotly_chart(fig_rc, use_container_width=True)

    # Үздік кандидаттар
    st.markdown('<p class="section-title">Үздік кандидаттар тізімі (Шортлист)</p>', unsafe_allow_html=True)
    engine = ScoringEngine()
    top_n  = st.slider("Кандидаттар саны", 5, 100, 20)
    sl_df  = engine.shortlist(
        df, top_n=top_n,
        region=None if sel_region == "Барлығы" else sel_region,
        direction=None if sel_dir == "Барлығы" else sel_dir,
    )

    if len(sl_df) > 0:
        st.dataframe(sl_df, use_container_width=True, height=380)
        csv_data = sl_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("CSV форматында жүктеу", csv_data,
                           "shortlist.csv", "text/csv")
    else:
        st.info("Таңдалған сүзгілер бойынша белсенді өтінімдер табылмады")

    # Жоғары тәуекел өтінімдері
    if "Risk_Level" in filtered.columns:
        high_df = filtered[filtered["Risk_Level"] == "High Risk"]
        if len(high_df) > 0:
            st.markdown(
                f'<p class="section-title">Жоғары тәуекел өтінімдері — {len(high_df):,} дана</p>',
                unsafe_allow_html=True)
            cols_show = [c for c in [
                "Номер заявки", "Область", "Направление водства",
                "Причитающая сумма", "Норматив", "Кол_голов",
                "Risk_Score", "Risk_Reasons", "Merit_Score",
            ] if c in high_df.columns]
            st.dataframe(high_df[cols_show].head(200), use_container_width=True, height=340)


# ═══════════════════════════════════════════════════════════════
# БЕТ 4: БОЛЖАМДАУ
# ═══════════════════════════════════════════════════════════════
elif page == "Болжамдау":
    st.markdown("""
    <div class="page-header">
        <p class="page-title">Болжамдау жүйесі</p>
        <p class="page-subtitle">Exponential Smoothing алгоритмі · Уақыт сериясы талдауы</p>
    </div>
    """, unsafe_allow_html=True)

    if "Дата поступления" not in df.columns:
        st.warning("Деректерде 'Дата поступления' бағаны жоқ")
        st.stop()

    f1, f2, f3 = st.columns(3)
    with f1:
        fc_region = st.selectbox("Облыс", ["Барлығы"] + sorted(df["Область"].unique().tolist()))
    with f2:
        fc_dir = st.selectbox("Бағыт",
                              ["Барлығы"] + sorted(df["Направление водства"].unique().tolist())
                              if "Направление водства" in df.columns else ["Барлығы"])
    with f3:
        fc_periods = st.slider("Болжам мерзімі (ай)", 1, 6, 3)

    ts = prepare_ts(
        df,
        region=None if fc_region == "Барлығы" else fc_region,
        direction=None if fc_dir == "Барлығы" else fc_dir,
    )

    if len(ts) < 2:
        st.warning("Болжам үшін деректер жеткіліксіз (кемінде 2 ай қажет)")
    else:
        fc_apps = forecast_series(ts, col="Заявок", periods=fc_periods)
        hist    = fc_apps[~fc_apps["is_forecast"]]
        fcast   = fc_apps[fc_apps["is_forecast"]]

        st.markdown('<p class="section-title">Өтінімдер саны болжамы</p>', unsafe_allow_html=True)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=hist["date"], y=hist["Заявок"],
            mode="lines+markers", name="Нақты деректер",
            line=dict(color="#1a6b3c", width=2.5),
            marker=dict(size=6),
        ))
        fig_fc.add_trace(go.Scatter(
            x=fcast["date"], y=fcast["Заявок"],
            mode="lines+markers", name="Болжам",
            line=dict(color="#d97706", width=2.5, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
        ))
        fig_fc.update_layout(**make_chart_layout(360),
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.1))
        fig_fc.update_layout(xaxis_title="Ай", yaxis_title="Өтінімдер саны")
        st.plotly_chart(fig_fc, use_container_width=True)

        if len(fcast) > 0:
            c1, c2, c3 = st.columns(3)
            for i, (_, row) in enumerate(fcast.iterrows()):
                [c1, c2, c3][i % 3].metric(
                    f"{row['date'].strftime('%Y-%m')}",
                    f"{int(row['Заявок'])} өтінім"
                )

        # Сомасы болжамы
        st.markdown('<p class="section-title">Субсидия сомасы болжамы</p>', unsafe_allow_html=True)
        fc_sum   = forecast_series(ts, col="Сумма", periods=fc_periods)
        hist_s   = fc_sum[~fc_sum["is_forecast"]]
        fcast_s  = fc_sum[fc_sum["is_forecast"]]

        fig_sum = go.Figure()
        fig_sum.add_trace(go.Bar(x=hist_s["date"], y=hist_s["Сумма"]/1e6,
                                 name="Нақты", marker_color="#1a6b3c", opacity=0.85))
        fig_sum.add_trace(go.Bar(x=fcast_s["date"], y=fcast_s["Сумма"]/1e6,
                                 name="Болжам", marker_color="#d97706", opacity=0.75))
        fig_sum.update_layout(**make_chart_layout(300), barmode="overlay",
                               xaxis_title="Ай", yaxis_title="Сомасы (млн ₸)")
        st.plotly_chart(fig_sum, use_container_width=True)

    # Барлық облыстар болжамы
    st.markdown('<p class="section-title">Барлық облыстар бойынша болжам кестесі</p>', unsafe_allow_html=True)
    with st.spinner("Есептелуде..."):
        reg_fc = regional_forecast(df, periods=3)
    if len(reg_fc) > 0:
        reg_fc.columns = ["Облыс", "Ағымдағы ай", "1 айдан кейін", "3 айдан кейін", "Тренд"]
        st.dataframe(reg_fc, use_container_width=True, height=480)

    # Бағыттар динамикасы
    if "Направление водства" in df.columns:
        st.markdown('<p class="section-title">Субсидия бағыттары динамикасы</p>', unsafe_allow_html=True)
        dir_tr = direction_trends(df)
        if len(dir_tr) > 0:
            fig_d = px.line(dir_tr, x="date", y="Заявок",
                            color="Направление водства",
                            color_discrete_sequence=px.colors.qualitative.Set2)
            fig_d.update_layout(**make_chart_layout(360),
                                xaxis_title="Ай", yaxis_title="Өтінімдер саны",
                                legend_title="Бағыт")
            st.plotly_chart(fig_d, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# БЕТ 5: ЖЕР ТАЛДАУЫ
# ═══════════════════════════════════════════════════════════════
elif page == "Жер талдауы":
    st.markdown("""
    <div class="page-header">
        <p class="page-title">Жер учаскесін талдау</p>
        <p class="page-subtitle">EfficientNet-B0 · YOLOv8 · Топырақ жіктеуі · Егін ұсынысы</p>
    </div>
    """, unsafe_allow_html=True)

    # Нұсқаулар
    st.markdown("""
    <div class="agro-card agro-card-blue">
        <h4 style="margin:0 0 10px; color:#1d4ed8;">📋 Фото жүктеу нұсқаулары</h4>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; font-size:13px; color:#374151;">
            <div>✅ Жерден <strong>30–80 см</strong> биіктікте түсіріңіз</div>
            <div>✅ Тек топырақ көрінетіндей кадрлаңыз</div>
            <div>✅ Жақсы жарықтықта түсіріңіз</div>
            <div>✅ <strong>2–5 фото</strong> жүктесеңіз нәтиже дәлірек болады</div>
            <div>❌ Алыстан панорама түсірмеңіз</div>
            <div>❌ Аспан, тас немесе өсімдік толтырмасын</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Облыс таңдау
    st.markdown('<p class="section-title">1. Облысты таңдаңыз</p>', unsafe_allow_html=True)
    sel_region_soil = st.selectbox(
        "Жер қай облыста орналасқан?",
        REGION_LIST,
        index=(REGION_LIST.index("Северо-Казахстанская область")
               if "Северо-Казахстанская область" in REGION_LIST else 0),
        label_visibility="collapsed",
    )

    # Фото жүктеу
    st.markdown('<p class="section-title">2. Жер суреттерін жүктеңіз</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Суреттерді таңдаңыз (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        images = []
        cols = st.columns(min(len(uploaded_files), 4))
        for i, uf in enumerate(uploaded_files):
            img = Image.open(uf)
            images.append(img)
            with cols[i % 4]:
                st.image(img, caption=f"Фото {i+1}", use_container_width=True)

        st.markdown('<p class="section-title">3. Талдауды іске қосыңыз</p>',
                    unsafe_allow_html=True)

        if st.button("Топырақ талдауын бастау", use_container_width=True):
            with st.spinner("🧠 AI талдауда..."):
                result = analyze_photos(images, sel_region_soil)

            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                st.markdown("""
                <p class="section-title">Талдау нәтижелері</p>
                """, unsafe_allow_html=True)

                # Жалпы баға
                st.markdown(f"""
                <div class="agro-card" style="border-left:5px solid {result['verdict_color']};
                     text-align:center; padding:32px;">
                    <div style="font-size:32px; font-weight:800;
                         color:{result['verdict_color']}; letter-spacing:-0.5px;">
                        {result['verdict']}
                    </div>
                    <div style="color:#6b7280; margin-top:6px; font-size:14px;">
                        {result['verdict_ru']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Метрикалар
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Топырақ түрі", result["soil_kz"])
                r2.metric("Сенімділік", f"{result['confidence']:.0f}%")
                r3.metric("Сапа рейтингі", f"{result['rating']}/5")
                r4.metric("Талданған фото", result["n_photos"])

                col_l, col_r = st.columns(2)

                with col_l:
                    # Топырақ ақпараты
                    st.markdown(f"""
                    <div class="agro-card agro-card-green">
                        <h4 style="margin:0 0 8px; color:#1a6b3c;">Топырақ сипаттамасы</h4>
                        <div style="font-size:15px; font-weight:600; color:#111827;">
                            {result['soil_kz']}
                        </div>
                        <div style="font-size:13px; color:#6b7280; margin-top:4px;">
                            {result['soil_ru']}
                        </div>
                        <div style="font-size:11px; color:#9ca3af; margin-top:8px;">
                            Үлгі: {result['model_used']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Ұсынылатын дақылдар
                    crops_html = " &nbsp;&nbsp; ".join(
                        [f"<span style='background:#f0fdf4; color:#166534; padding:4px 10px; "
                         f"border-radius:6px; font-size:13px; font-weight:500;'>🌾 {c}</span>"
                         for c in result["recommended_crops"]]
                    )
                    st.markdown(f"""
                    <div class="agro-card agro-card-green">
                        <h4 style="margin:0 0 12px; color:#1a6b3c;">
                            Ұсынылатын дақылдар — {sel_region_soil.split(' ')[0]}
                        </h4>
                        <div style="line-height:2.2;">{crops_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_r:
                    # YOLO проблемалары
                    if result["problems_found"]:
                        problems_html = "".join([
                            f"<div style='padding:6px 0; border-bottom:1px solid #fef3c7; "
                            f"font-size:13px;'>{k} — {v} рет анықталды</div>"
                            for k, v in result["problems"].items()
                        ])
                        st.markdown(f"""
                        <div class="agro-card agro-card-orange">
                            <h4 style="margin:0 0 10px; color:#d97706;">
                                Анықталған мәселелер
                            </h4>
                            {problems_html}
                        </div>
                        """, unsafe_allow_html=True)

                    # Ұсыныстар
                    recs_html = "".join([
                        f"<div style='padding:5px 0; font-size:13px; "
                        f"border-bottom:1px solid #f3f4f6;'>{r}</div>"
                        for r in result["recommendations"]
                    ])
                    st.markdown(f"""
                    <div class="agro-card">
                        <h4 style="margin:0 0 10px; color:#374151;">Агрономдық ұсыныстар</h4>
                        {recs_html}
                    </div>
                    """, unsafe_allow_html=True)

                    # Сәйкестік
                    c = result["consistency"]
                    c_color = "#1a6b3c" if c >= 70 else "#d97706"
                    c_text  = ("Фотолар бір-біріне сәйкес — нәтиже сенімді"
                               if c >= 70 else
                               "Фотолардағы жер жағдайы әртүрлі — бөлек тексеру ұсынылады")
                    st.markdown(f"""
                    <div class="agro-card">
                        <h4 style="margin:0 0 6px; color:#374151;">
                            Фотолар сәйкестігі: 
                            <span style="color:{c_color};">{c:.0f}%</span>
                        </h4>
                        <div style="font-size:13px; color:#6b7280;">{c_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Топырақ скор диаграммасы
                st.markdown('<p class="section-title">Топырақ түрлері бойынша сәйкестік</p>',
                            unsafe_allow_html=True)
                scores_df = pd.DataFrame({
                    "Топырақ түрі": list(result["all_scores"].keys()),
                    "Ықтималдық (%)": list(result["all_scores"].values()),
                }).sort_values("Ықтималдық (%)", ascending=True)

                fig_s = px.bar(scores_df, x="Ықтималдық (%)", y="Топырақ түрі",
                               orientation="h",
                               color="Ықтималдық (%)",
                               color_continuous_scale=["#e5e7eb","#1a6b3c"])
                fig_s.update_layout(**make_chart_layout(320), coloraxis_showscale=False)
                fig_s.update_layout(yaxis_title="", xaxis_title="Ықтималдық (%)")
                st.plotly_chart(fig_s, use_container_width=True)

    else:
        # Анықтамалық
        st.markdown('<p class="section-title">Топырақ түрлері анықтамалығы</p>',
                    unsafe_allow_html=True)
        cols_ref = st.columns(2)
        for i, (soil_name, info) in enumerate(SOIL_INFO.items()):
            with cols_ref[i % 2]:
                stars = "⭐" * info["rating"]
                color = "#1a6b3c" if info["rating"] >= 4 else \
                        "#d97706" if info["rating"] == 3 else "#dc2626"
                st.markdown(f"""
                <div class="agro-card" style="border-left:4px solid {color};">
                    <div style="font-weight:700; color:#111827; font-size:14px;">
                        {info.get('kz', soil_name)}
                    </div>
                    <div style="font-size:12px; color:#6b7280; margin:4px 0;">
                        {info.get('ru', '')}
                    </div>
                    <div style="font-size:13px;">{stars}</div>
                </div>
                """, unsafe_allow_html=True)
