"""
AgroSmart KZ — Негізгі қосымша
Іске қосу: streamlit run app.py
"""

from __future__ import annotations

from subsidy_verifier import verify
import forecasting
from soil_analyzer import analyze_photos, SOIL_INFO
from scoring_engine import ScoringEngine
from config import KZ_REGIONS, REGION_LIST
from streamlit_folium import st_folium
from PIL import Image
import folium
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st
import pandas as pd
import os
import glob
import json
import time
import warnings
from urllib.parse import quote

warnings.filterwarnings("ignore")

pio.templates.default = "plotly_white"
px.defaults.template = "plotly_white"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─────────────────────────────────────────────────────────────
# БЕТ КОНФИГУРАЦИЯСЫ
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroSmart KZ",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS — КӘСІБИ ДИЗАЙН
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
.icon-asset {
    width: 18px;
    height: 18px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 auto;
    color: inherit;
}
.icon-asset svg {
    width: 100%;
    height: 100%;
    display: block;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.9;
    stroke-linecap: round;
    stroke-linejoin: round;
}
:root {
    --bg: #f6f8f7;
    --surface: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --line: #e5ebe7;
    --green: #0f8a45;
    --green-2: #16a34a;
    --green-soft: #eaf8ef;
}
html, body { background: var(--bg) !important; color: var(--text) !important; }
.stApp {
    background:
      radial-gradient(1200px 600px at 110% -20%, rgba(15,138,69,0.05), transparent 38%),
      var(--bg) !important;
}
.main .block-container {
    padding: 5.2rem 2rem 2.4rem !important;
    max-width: 1360px;
}
button[data-testid="collapsedControl"], button[aria-label*="Collapse sidebar"], button[aria-label*="Expand sidebar"] {
    display: none !important;
}
header[data-testid="stHeader"],
footer,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
div[data-testid="stRuntimeMessage"],
div[data-testid="stElementToolbar"],
div[data-testid="stElementToolbarButton"] {
    display: none !important;
}
button[data-testid="stDeployButton"],
div[data-testid="stDeployButton"] {
    display: none !important;
}
.material-symbols-outlined,
.material-symbols-rounded,
.material-symbols-sharp {
    display: none !important;
}
span[class*="material"],
span[class*="Material"] {
    display: none !important;
}
span[style*="Material Symbols"],
span[style*="material symbols"],
span[aria-hidden="true"],
i.material-icons,
i.material-icons-outlined,
i.material-icons-round,
i.material-icons-sharp {
    display: none !important;
}

.top-shell {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    z-index: 999;
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--line);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
}
.top-brand {
    font-size: 17px;
    font-weight: 800;
    letter-spacing: -0.3px;
    color: #07250f;
}
.top-chip {
    border: 1px solid #b7e7c8;
    background: #effcf3;
    color: #0a6d35;
    font-size: 11px;
    font-weight: 700;
    border-radius: 999px;
    padding: 4px 10px;
}
.top-page {
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
}
.top-actions {
    display: flex;
    align-items: center;
    gap: 10px;
}
.top-search {
    min-width: 220px;
    border: 1px solid var(--line);
    background: #f8fbf9;
    color: #94a3b8;
    font-size: 12px;
    padding: 7px 12px;
    border-radius: 999px;
}
.top-icon {
    width: 30px;
    height: 30px;
    border-radius: 999px;
    display: grid;
    place-items: center;
    border: 1px solid var(--line);
    background: #ffffff;
    color: #0f172a;
    font-size: 13px;
}
.sidebar-shell {
    padding: 4px 0 0;
}
.sidebar-brand {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.4px;
    color: #111827;
    line-height: 1.15;
}
.sidebar-sub {
    font-size: 11px;
    color: #6b7280;
    margin-top: 3px;
}
.sidebar-divider {
    height: 1px;
    background: #edf2ee;
    margin: 14px 0;
}
.sidebar-group-title {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    font-weight: 700;
    color: #94a3b8;
    margin: 0 0 10px;
}
.sidebar-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.sidebar-link {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    border-radius: 12px;
    padding: 11px 12px;
    background: #ffffff;
    border: 1px solid #eef2ef;
    color: #4b5563;
    text-decoration: none;
    font-size: 13px;
    font-weight: 600;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
    transition: all 0.15s ease;
}
.sidebar-link:hover {
    background: #f6fbf8;
    border-color: #dce8e0;
    color: #0f172a;
}
.sidebar-link.active {
    background: #e8f8ee;
    border-color: #d3eadc;
    color: #0f7d43;
}
.ranking-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.ranking-item {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
}
.ranking-left {
    flex: 1 1 auto;
}
.ranking-title {
    font-size: 13px;
    font-weight: 700;
    color: #0f172a;
}
.ranking-sub {
    margin-top: 4px;
    font-size: 11px;
    color: #64748b;
}
.ranking-bar {
    margin-top: 8px;
    height: 6px;
    border-radius: 999px;
    background: #eef2ef;
    overflow: hidden;
}
.ranking-bar span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #0f8a45, #7ce6a6);
}
.ranking-value {
    font-size: 12px;
    font-weight: 700;
    color: #0f8a45;
    min-width: 54px;
    text-align: right;
}
.sidebar-footer {
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px solid #edf2ee;
}
.sidebar-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    border-radius: 10px;
    padding: 11px 12px;
    text-decoration: none;
    font-size: 13px;
    font-weight: 600;
}
.sidebar-btn.primary {
    background: #0f8a45;
    color: #ffffff !important;
    box-shadow: 0 8px 16px rgba(15,138,69,0.18);
}
.sidebar-btn.primary .icon-asset { color: #ffffff !important; }
.sidebar-btn.primary span { color: #ffffff !important; }
.sidebar-btn.primary svg { stroke: #ffffff !important; }
.sidebar-btn.secondary {
    margin-top: 10px;
    background: transparent;
    color: #475569;
}
.sidebar-btn.danger {
    color: #ef4444;
    margin-top: 6px;
}
.surface-panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    box-shadow: 0 5px 14px rgba(15, 23, 42, 0.05);
    padding: 14px 14px;
    margin-bottom: 12px;
}
.dashboard-kpi {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 3px 10px rgba(15,23,42,0.05);
    min-height: 112px;
}
.dashboard-kpi .label {
    text-transform: uppercase;
    font-size: 10px;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: .6px;
}
.dashboard-kpi .value {
    margin-top: 8px;
    font-size: 30px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}
.dashboard-kpi .note {
    margin-top: 8px;
    font-size: 12px;
    font-weight: 600;
    color: #16a34a;
}
.dashboard-kpi.warn .note { color: #d97706; }
.dashboard-kpi.risk .note { color: #dc2626; }
.insight-panel {
    background: #ffffff;
    color: var(--text);
    border-radius: 14px;
    border: 1px solid var(--line);
    padding: 14px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
}
.insight-panel h4 {
    font-size: 18px;
    line-height: 1.2;
    margin: 0 0 10px;
}
.insight-box {
    background: #f8fbf9;
    border: 1px solid #e3ebe6;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 10px;
}
.insight-box .ib-title {
    margin: 0 0 6px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .4px;
}
.insight-box .ib-text {
    margin: 0;
    font-size: 12px;
    line-height: 1.45;
    color: #475569;
}
.ai-card {
    background: linear-gradient(180deg, #0f6b3b 0%, #0a5b32 100%);
    color: #ffffff;
    border-radius: 16px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 10px 22px rgba(6, 64, 34, 0.2);
}
.ai-card .ai-title {
    font-size: 18px;
    font-weight: 800;
    margin: 0 0 6px;
}
.ai-card .ai-sub {
    font-size: 12px;
    color: rgba(255,255,255,0.85);
    margin: 0 0 12px;
}
.ai-list {
    display: grid;
    gap: 10px;
    margin-bottom: 12px;
}
.ai-item {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 10px 12px;
}
.ai-icon {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: rgba(255,255,255,0.12);
    display: grid;
    place-items: center;
    color: #ffffff;
}
.ai-icon svg {
    width: 18px;
    height: 18px;
    display: block;
    fill: #ffffff;
}
.ai-item-title {
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
}
.ai-item-sub {
    font-size: 11px;
    color: rgba(255,255,255,0.8);
}
.ai-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    background: #ffffff;
    color: #0f6b3b;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 700;
    text-decoration: none;
}
.section-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.section-head .title {
    margin: 0;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #0f172a;
}
.section-head .sub {
    margin: 4px 0 0;
    font-size: 13px;
    color: #64748b;
}
.ghost-filters {
    display: flex;
    gap: 8px;
}
.ghost-filters .f {
    border: 1px solid var(--line);
    background: #fff;
    border-radius: 10px;
    padding: 7px 11px;
    font-size: 12px;
    color: #334155;
}

.soil-chip {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: #e7f7ee;
    border: 1px solid #c8e8d6;
    color: #0b6d36;
    font-size: 11px;
    font-weight: 700;
}
.soil-grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 14px;
}
.soil-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    padding: 16px;
}
.soil-card h3 {
    margin: 0;
    font-size: 19px;
    color: #0f172a;
}
.soil-card h4 {
    margin: 0;
    font-size: 14px;
    color: #1f2937;
}
.soil-muted {
    margin: 6px 0 0;
    color: #64748b;
    font-size: 12px;
}
.soil-upload-zone {
    margin-top: 10px;
    padding: 20px;
    border: 2px dashed #c5e5d2;
    border-radius: 12px;
    background: linear-gradient(180deg, #fbfffc, #f2faf5);
    text-align: center;
}
.soil-kpi {
    background: #f8fbf9;
    border: 1px solid #dfe9e3;
    border-radius: 12px;
    padding: 10px;
}
.soil-kpi .k {
    margin: 0;
    text-transform: uppercase;
    letter-spacing: .4px;
    font-size: 10px;
    color: #94a3b8;
    font-weight: 700;
}
.soil-kpi .v {
    margin: 5px 0 0;
    font-size: 17px;
    font-weight: 800;
    color: #0f172a;
}
.soil-right {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 16px;
    color: var(--text);
    padding: 14px;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.06);
}
.soil-right h3 {
    color: var(--text);
}
.soil-right-box {
    margin-top: 10px;
    background: #f8fbf9;
    border: 1px solid #e3ebe6;
    border-radius: 10px;
    padding: 10px;
}
.soil-right-box p {
    margin: 0;
    font-size: 12px;
    color: #475569;
    line-height: 1.45;
}
.soil-history-item {
    border-left: 3px solid #9bd9b3;
    padding-left: 10px;
    margin-bottom: 9px;
}

section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--line) !important;
    box-shadow: 2px 0 10px rgba(0,0,0,0.03);
    padding-top: 18px !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stRadio > div { display: none !important; }
section[data-testid="stSidebar"] button[data-testid="collapsedControl"],
section[data-testid="stSidebar"] button[aria-label*="Collapse sidebar"],
section[data-testid="stSidebar"] button[aria-label*="Expand sidebar"] { display: none !important; }

div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 20px 22px 16px !important;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s, transform 0.2s;
}
div[data-testid="stMetric"]:hover {
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.09);
    transform: translateY(-1px);
}
div[data-testid="stMetric"]::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--green), var(--green-2));
    border-radius: 0 0 16px 16px;
}
div[data-testid="stMetric"] label {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
div[data-testid="stMetric"] > div > div {
    font-size: 26px !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    letter-spacing: -0.5px;
}

div.stButton > button {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    padding: 11px 26px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    box-shadow: 0 7px 18px rgba(15, 23, 42, 0.08) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    box-shadow: 0 10px 20px rgba(15, 23, 42, 0.12) !important;
    transform: translateY(-1px) !important;
}

.agro-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 14px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
    transition: box-shadow 0.2s;
}
.agro-card:hover { box-shadow: 0 8px 18px rgba(15, 23, 42, 0.09); }
.agro-card-green  { border-top: 3px solid var(--green-2); }
.agro-card-red    { border-top: 3px solid #dc2626; }
.agro-card-orange { border-top: 3px solid #d97706; }
.agro-card-blue   { border-top: 3px solid #2563eb; }

.page-header {
    margin-bottom: 24px;
    padding-bottom: 14px;
    border-bottom: 1px solid #e8edf2;
}
.page-title {
    font-size: 30px;
    font-weight: 800;
    color: var(--text);
    margin: 0 0 4px;
    letter-spacing: -0.8px;
}
.page-subtitle {
    font-size: 14px;
    color: #6b7280;
    margin: 0;
    font-weight: 500;
}

.section-title {
    font-size: 14px;
    font-weight: 700;
    color: #1f2937;
    margin: 24px 0 12px;
    letter-spacing: 0.2px;
}

.info-box {
    background: #eefdF3;
    border: 1px solid #bae7ca;
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 13px;
    color: #0b6b34;
    font-weight: 600;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    border: 1px solid var(--line) !important;
    overflow: hidden;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    background: #ffffff !important;
}
div[data-testid="stDataFrame"] * {
    color: #0f172a !important;
}
div[data-testid="stDataFrame"] div[role="columnheader"],
div[data-testid="stDataFrame"] div[role="gridcell"] {
    background: #ffffff !important;
}
div[data-testid="stDataFrame"] div[role="row"] {
    background: #ffffff !important;
}
div[data-testid="stDataFrame"] div[data-testid="stDataFrameResizable"] {
    background: #ffffff !important;
}
div[data-testid="stDataFrame"] table {
    background: #ffffff !important;
}

hr { border-color: #e8edf2 !important; margin: 20px 0 !important; }

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stMultiSelect"] div[role="combobox"],
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input,
div[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
}
div[data-testid="stSelectbox"] svg,
div[data-testid="stMultiSelect"] svg {
    color: #0f172a !important;
}
div[data-baseweb="select"] {
    background: #ffffff !important;
    color: #0f172a !important;
}
div[data-baseweb="select"] * {
    color: #0f172a !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div,
div[data-baseweb="select"] [data-baseweb="input"],
div[data-baseweb="select"] [data-baseweb="input"] > div,
div[data-baseweb="select"] [data-baseweb="input"] input {
    background: #ffffff !important;
    color: #0f172a !important;
}
div[data-baseweb="select"] [data-baseweb="input"] svg,
div[data-baseweb="select"] svg {
    color: #0f172a !important;
}
div[data-baseweb="popover"],
div[data-baseweb="menu"],
div[data-baseweb="menu"] * {
    background: #ffffff !important;
    color: #0f172a !important;
}
div[data-baseweb="menu"] li[role="option"] {
    background: #ffffff !important;
}
div[data-baseweb="menu"] li[role="option"][aria-selected="true"],
div[data-baseweb="menu"] li[role="option"]:hover {
    background: #f1f5f9 !important;
}
div[data-baseweb="select"] input {
    background: #ffffff !important;
    color: #0f172a !important;
}
div[data-baseweb="select"] [role="listbox"],
div[data-baseweb="select"] [role="option"],
div[data-baseweb="select"] [role="option"] * {
    background: #ffffff !important;
    color: #0f172a !important;
}
div[data-baseweb="select"] [role="option"][aria-selected="true"],
div[data-baseweb="select"] [role="option"]:hover {
    background: #f1f5f9 !important;
}
div[data-baseweb="popover"] [role="listbox"] {
    background: #ffffff !important;
    color: #0f172a !important;
}
div[data-baseweb="popover"] [role="listbox"] * {
    color: #0f172a !important;
}
div[data-testid="stHorizontalBlock"]:empty,
div[data-testid="stVerticalBlock"]:empty,
div[data-testid="stColumn"]:empty,
div[data-testid="element-container"]:empty {
    display: none !important;
}
div[role="listbox"],
div[role="listbox"] * {
    background: #ffffff !important;
    color: #0f172a !important;
}
li[role="option"],
li[role="option"] * {
    background: #ffffff !important;
    color: #0f172a !important;
}
li[role="option"][aria-selected="true"],
li[role="option"]:hover {
    background: #f1f5f9 !important;
}
div[data-testid="stSelectbox"] [role="listbox"],
div[data-testid="stMultiSelect"] [role="listbox"] {
    background: #ffffff !important;
    color: #0f172a !important;
}

@media (max-width: 1200px) {
    .main .block-container { padding: 5.2rem 1rem 2rem !important; }
    .top-search { display: none; }
    .soil-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# ДЕРЕКТЕРДІ ЖҮКТЕУ
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Деректер жүктелуде...")
def load_data() -> pd.DataFrame | None:
    csv_path = os.path.join(RESULTS_DIR, "processed_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        if "Дата поступления" in df.columns:
            df["Дата поступления"] = pd.to_datetime(
                df["Дата поступления"], errors="coerce"
            )
        return df

    xlsx_files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
    if not xlsx_files:
        return None

    xlsx_path = max(xlsx_files, key=os.path.getsize)
    try:
        raw = pd.read_excel(xlsx_path, skiprows=4)
    except Exception:
        raw = pd.read_excel(xlsx_path)

    engine = ScoringEngine()
    df = engine.run(raw)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    engine.save(MODELS_DIR)
    return df


def icon_svg(name: str) -> str:
    icons = {
        "dashboard": '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="7" height="7" rx="1.4"></rect><rect x="14" y="3" width="7" height="7" rx="1.4"></rect><rect x="3" y="14" width="7" height="7" rx="1.4"></rect><rect x="14" y="14" width="7" height="7" rx="1.4"></rect></svg>',
        "map": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6l6-2 4 2 6-2v14l-6 2-4-2-6 2z"></path><path d="M10 4v16"></path><path d="M14 6v16"></path></svg>',
        "psychology": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18h6"></path><path d="M8 14c-1.8-1.2-3-3.2-3-5.5A6.5 6.5 0 0 1 17.5 7c0 1.8-.7 3.3-1.8 4.5-.9 1-1.7 1.7-2 3.5"></path><circle cx="10.5" cy="9" r="2.2"></circle></svg>',
        "query_stats": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19h16"></path><path d="M6 16l4-4 3 3 5-6"></path><path d="M18 9v4h-4"></path></svg>',
        "grass": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 20c2.4-4.8 5.2-7.4 9-9"></path><path d="M10 20c.7-3.8 2.5-7.2 6-11"></path><path d="M15 20c1.2-2.8 2.8-5.2 4-7"></path><path d="M4 20h16"></path></svg>',
        "verified_user": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l7 3v5c0 4.8-2.9 8.2-7 10-4.1-1.8-7-5.2-7-10V6z"></path><path d="M9.2 12.2l2 2.1 3.8-4.3"></path></svg>',
        "download": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v9"></path><path d="M8.5 10.5L12 14l3.5-3.5"></path><path d="M5 20h14"></path></svg>',
        "settings": '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.2"></circle><path d="M12 4.5v2.1"></path><path d="M12 17.4v2.1"></path><path d="M4.5 12h2.1"></path><path d="M17.4 12h2.1"></path><path d="M6.2 6.2l1.5 1.5"></path><path d="M16.3 16.3l1.5 1.5"></path><path d="M17.8 6.2l-1.5 1.5"></path><path d="M7.7 16.3l-1.5 1.5"></path></svg>',
        "logout": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 17l-1.5 0A2.5 2.5 0 0 1 6 14.5v-5A2.5 2.5 0 0 1 8.5 7H10"></path><path d="M14 8l4 4-4 4"></path><path d="M18 12H10"></path></svg>',
        "notifications": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 17H9"></path><path d="M6.5 17h11l-1.2-1.6c-.5-.7-.8-1.6-.8-2.5V10a3.5 3.5 0 0 0-7 0v2.9c0 .9-.3 1.8-.8 2.5z"></path><path d="M11 18.5a1 1 0 0 0 2 0"></path></svg>',
        "eco": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18c4.5-8.5 9.5-12 16-13-1 6.5-4.5 11.5-13 16"></path><path d="M10 14c1.6 0 3-.8 4-2"></path></svg>',
    }
    return f'<span class="icon-asset">{icons.get(name, icons["settings"])}</span>'


# ─────────────────────────────────────────────────────────────
# БҮЙІР ПАНЕЛІ
# ─────────────────────────────────────────────────────────────
nav_items = [
    ("Жалпы шолу", "dashboard"),
    ("Аймақтық карта", "map"),
    ("AI Скоринг", "psychology"),
    ("Болжамдау", "query_stats"),
    ("Жер талдауы", "grass"),
    ("Субсидия верификациясы", "verified_user"),
]

page = st.query_params.get("page", "Жалпы шолу")
if page not in [label for label, _ in nav_items]:
    page = "Жалпы шолу"

with st.sidebar:
    st.markdown(
        """
    <div class="sidebar-shell">
        <div class="sidebar-brand">AgroSmart KZ</div>
        <div class="sidebar-sub">Мемлекеттік портал</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-group-title">Навигация</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-list">', unsafe_allow_html=True)
    for label, icon_name in nav_items:
        active_class = "active" if label == page else ""
        st.markdown(
            f'<a class="sidebar-link {active_class}" href="?page={quote(label)}">{icon_svg(icon_name)}<span>{label}</span></a>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
    st.markdown(f'<a class="sidebar-btn primary" href="#">{icon_svg("download")}<span>Есепті жүктеу</span></a>', unsafe_allow_html=True)
    st.markdown(f'<a class="sidebar-btn secondary" href="#">{icon_svg("settings")}<span>Параметрлер</span></a>', unsafe_allow_html=True)
    st.markdown(f'<a class="sidebar-btn danger" href="#">{icon_svg("logout")}<span>Шығу</span></a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)



st.markdown(
    f"""
<div class="top-shell">
    <div style="display:flex; align-items:center; gap:10px;">
        <div class="top-brand">AgroSmart KZ</div>
        <span class="top-chip">Ақылды платформа</span>
    </div>
    <div class="top-actions">
        <div class="top-search">Іздеу</div>
        <div class="top-icon">{icon_svg("settings")}</div>
        <div class="top-icon">{icon_svg("notifications")}</div>
        <div class="top-page">Бөлім: {page}</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# ДЕРЕКТЕРДІ ЖҮКТЕУ
# ─────────────────────────────────────────────────────────────
df = load_data()

if df is None:
    st.markdown(
        """
    <div class="agro-card agro-card-red">
        <h3 style="color:#dc2626; margin:0 0 8px;">Деректер табылмады</h3>
        <p style="color:#6b7280; margin:0;">
            1. <code>data/</code> папкасына xlsx файлын қойыңыз<br>
            2. <code>python train_models.py</code> іске қосыңыз<br>
            3. <code>streamlit run app.py</code>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

# Жалпы көрсеткіштер
total_apps = len(df)
total_sum = df["Причитающая сумма"].sum() if "Причитающая сумма" in df.columns else 0
high_risk_n = (
    int((df["Risk_Level"] == "High Risk").sum()) if "Risk_Level" in df.columns else 0
)
exec_pct = (
    round((df["Статус заявки"] == "Исполнена").sum() / total_apps * 100, 1)
    if total_apps
    else 0
)
avg_merit = round(df["Merit_Score"].mean(), 1) if "Merit_Score" in df.columns else 0


def make_chart_layout(height=340):
    return dict(
        height=height,
        margin=dict(t=36, b=10, l=10, r=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter", size=12, color="#374151"),
        template="plotly_white",
        legend=dict(font=dict(color="#0f172a")),
        xaxis=dict(
            tickfont=dict(color="#0f172a"),
            title=dict(font=dict(color="#0f172a")),
            gridcolor="#e5ebe7",
        ),
        yaxis=dict(
            tickfont=dict(color="#0f172a"),
            title=dict(font=dict(color="#0f172a")),
            gridcolor="#e5ebe7",
        ),
    )


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def format_tenge_short(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.1f} млрд ₸"
    if value >= 1e6:
        return f"{value / 1e6:.1f} млн ₸"
    return f"{value:,.0f} ₸"


def style_table(df: pd.DataFrame):
    return (
        df.style.set_properties(
            **{
                "background-color": "#ffffff",
                "color": "#0f172a",
                "border-color": "#e5ebe7",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#ffffff"),
                        ("color", "#0f172a"),
                        ("border-color", "#e5ebe7"),
                        ("font-weight", "700"),
                    ],
                }
            ]
        )
    )


# ═══════════════════════════════════════════════════════════════
# БЕТ 1: ЖАЛПЫ ШОЛУ
# ═══════════════════════════════════════════════════════════════
if page == "Жалпы шолу":
    st.markdown(
        """
    <div class="section-head">
        <div>
            <p class="title">Жалпы шолу</p>
            <p class="sub">Субсидия өтінімдерінің жиынтық аналитикасы · 2025 жыл</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f"""
    <div class="dashboard-kpi">
        <div class="label">Барлық өтінім</div>
        <div class="value">{total_apps:,}</div>
        <div class="note">+ ағымдағы кезең бойынша</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    k2.markdown(
        f"""
    <div class="dashboard-kpi">
        <div class="label">Жалпы сомасы</div>
        <div class="value">{total_sum / 1e9:.2f} млрд ₸</div>
        <div class="note">+ бюджет динамикасы</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"""
    <div class="dashboard-kpi warn">
        <div class="label">Орындалу үлесі</div>
        <div class="value">{exec_pct}%</div>
        <div class="note">мерзімдік орындалу</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    k4.markdown(
        f"""
    <div class="dashboard-kpi risk">
        <div class="label">Жоғары тәуекел / Merit</div>
        <div class="value">{high_risk_n:,} / {avg_merit:.1f}</div>
        <div class="note">тәуекел мониторингі</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 1])

    with left:
        c1, c2 = st.columns([1.15, 1])

        with c1:
            if "Статус заявки" in df.columns:
                st.markdown('<div class="surface-panel">', unsafe_allow_html=True)
                st.markdown(
                    '<p class="section-title" style="margin-top:0;">Өтінімдер статусы бойынша</p>',
                    unsafe_allow_html=True,
                )
                status_counts = df["Статус заявки"].value_counts().reset_index()
                status_counts.columns = ["Статус", "Саны"]
                fig = px.pie(
                    status_counts,
                    names="Статус",
                    values="Саны",
                    color_discrete_sequence=[
                        "#0f8a45",
                        "#22c55e",
                        "#86efac",
                        "#f59e0b",
                        "#ef4444",
                    ],
                    hole=0.55,
                )
                fig.update_layout(**make_chart_layout(330))
                fig.update_traces(textposition="outside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            if "Направление водства" in df.columns:
                st.markdown('<div class="surface-panel">', unsafe_allow_html=True)
                st.markdown(
                    '<p class="section-title" style="margin-top:0;">Субсидия бағыттары (ТОП-10)</p>',
                    unsafe_allow_html=True,
                )
                dir_cnt = df["Направление водства"].value_counts().head(10).reset_index()
                dir_cnt.columns = ["Бағыт", "Саны"]
                fig2 = px.bar(
                    dir_cnt,
                    x="Саны",
                    y="Бағыт",
                    orientation="h",
                    color="Саны",
                    color_continuous_scale=["#c7f1d8", "#0f8a45"],
                )
                fig2.update_layout(**make_chart_layout(330), coloraxis_showscale=False)
                fig2.update_layout(yaxis_title="", xaxis_title="Өтінімдер саны")
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if "Дата поступления" in df.columns:
            st.markdown('<div class="surface-panel">', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-title" style="margin-top:0;">Өтінімдер динамикасы (ай бойынша)</p>',
                unsafe_allow_html=True,
            )
            ts_all = forecasting.prepare_ts(df)
            if len(ts_all) > 0:
                fig3 = go.Figure()
                fig3.add_trace(
                    go.Scatter(
                        x=ts_all["date"],
                        y=ts_all["Заявок"],
                        fill="tozeroy",
                        line=dict(color="#0f8a45", width=2.5),
                        fillcolor="rgba(15,138,69,0.11)",
                        name="Өтінімдер саны",
                    )
                )
                fig3.update_layout(**make_chart_layout(280))
                fig3.update_layout(
                    xaxis_title="Ай", yaxis_title="Өтінімдер саны", showlegend=False
                )
                st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        rank_left, rank_right = st.columns(2)
        region_col = resolve_column(df, ["Область", "Регион", "Region", "Аймақ"])
        direction_col = resolve_column(
            df, ["Направление водства", "Направление", "Бағыт", "Субсидия бағыты"]
        )
        sum_col = "Причитающая сумма" if "Причитающая сумма" in df.columns else None

        with rank_left:
            st.markdown('<div class="surface-panel">', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-title" style="margin-top:0;">Өтінімдер бойынша рейтинг</p>',
                unsafe_allow_html=True,
            )
            if region_col:
                region_rank = (
                    df.groupby(region_col)
                    .agg(
                        Өтінімдер=(region_col, "size"),
                        Сома=(sum_col, "sum") if sum_col else (region_col, "size"),
                    )
                    .reset_index()
                )
                region_rank = region_rank.sort_values("Өтінімдер", ascending=False).head(5)
                max_apps = int(region_rank["Өтінімдер"].max()) if len(region_rank) else 1
                items = ["<div class=\"ranking-list\">"]
                for _, row in region_rank.iterrows():
                    pct = (row["Өтінімдер"] / max_apps * 100) if max_apps else 0
                    amount_label = (
                        format_tenge_short(row["Сома"]) if sum_col else ""
                    )
                    sub = (
                        f"{row['Өтінімдер']:,} өтінім · {amount_label}"
                        if sum_col
                        else f"{row['Өтінімдер']:,} өтінім"
                    )
                    items.append(
                        f"<div class=\"ranking-item\">"
                        f"<div class=\"ranking-left\">"
                        f"<div class=\"ranking-title\">{row[region_col]}</div>"
                        f"<div class=\"ranking-sub\">{sub}</div>"
                        f"<div class=\"ranking-bar\"><span style=\"width:{pct:.0f}%\"></span></div>"
                        f"</div>"
                        f"<div class=\"ranking-value\">{row['Өтінімдер']:,}</div>"
                        f"</div>"
                    )
                items.append("</div>")
                st.markdown("".join(items), unsafe_allow_html=True)
            else:
                st.caption("Аймақ атауы бар баған табылмады.")
            st.markdown('</div>', unsafe_allow_html=True)

        with rank_right:
            st.markdown('<div class="surface-panel">', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-title" style="margin-top:0;">Субсидия бағыттары бойынша рейтинг</p>',
                unsafe_allow_html=True,
            )
            if direction_col:
                if sum_col:
                    dir_rank = (
                        df.groupby(direction_col)
                        .agg(
                            Сома=(sum_col, "sum"),
                            Өтінімдер=(direction_col, "size"),
                        )
                        .reset_index()
                    )
                    dir_rank = dir_rank.sort_values("Сома", ascending=False).head(5)
                    max_sum = float(dir_rank["Сома"].max()) if len(dir_rank) else 1
                else:
                    dir_rank = (
                        df[direction_col]
                        .value_counts()
                        .head(5)
                        .reset_index()
                    )
                    dir_rank.columns = [direction_col, "Өтінімдер"]
                    dir_rank["Сома"] = dir_rank["Өтінімдер"].astype(float)
                    dir_rank["Өтінімдер"] = dir_rank["Өтінімдер"].astype(int)
                    max_sum = float(dir_rank["Сома"].max()) if len(dir_rank) else 1

                items = ["<div class=\"ranking-list\">"]
                for _, row in dir_rank.iterrows():
                    pct = (row["Сома"] / max_sum * 100) if max_sum else 0
                    amount_label = format_tenge_short(row["Сома"]) if sum_col else ""
                    sub = (
                        f"{row['Өтінімдер']:,} өтінім"
                        if sum_col
                        else f"{int(row['Сома']):,} өтінім"
                    )
                    value_label = amount_label if sum_col else f"{int(row['Сома']):,}"
                    items.append(
                        f"<div class=\"ranking-item\">"
                        f"<div class=\"ranking-left\">"
                        f"<div class=\"ranking-title\">{row[direction_col]}</div>"
                        f"<div class=\"ranking-sub\">{sub}</div>"
                        f"<div class=\"ranking-bar\"><span style=\"width:{pct:.0f}%\"></span></div>"
                        f"</div>"
                        f"<div class=\"ranking-value\">{value_label}</div>"
                        f"</div>"
                    )
                items.append("</div>")
                st.markdown("".join(items), unsafe_allow_html=True)
            else:
                st.caption("Бағыт атауы бар баған табылмады.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="surface-panel">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-title" style="margin-top:0;">Тәуекел деңгейі бойынша бөліну</p>',
            unsafe_allow_html=True,
        )
        if "Risk_Level" in df.columns:
            rc = df["Risk_Level"].value_counts().reset_index()
            rc.columns = ["Деңгей", "Саны"]
            cmap = {"Normal": "#0f8a45", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"}
            fig4 = px.bar(
                rc,
                x="Деңгей",
                y="Саны",
                color="Деңгей",
                color_discrete_map=cmap,
                text="Саны",
            )
            fig4.update_traces(textposition="outside")
            fig4.update_layout(**make_chart_layout(280), showlegend=False)
            fig4.update_layout(xaxis_title="", yaxis_title="Өтінімдер саны")
            st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        high_risk_pct = (high_risk_n / total_apps * 100) if total_apps else 0
        st.markdown(
            f"""
        <div class="ai-card">
            <div class="ai-title"><span style="display:inline-flex; align-items:center; gap:6px;"><svg viewBox="0 0 24 24" aria-hidden="true" style="width:16px;height:16px;fill:#ffffff;display:block;"><path d="M12 3l2.2 4.8L19 10l-4.8 2.2L12 17l-2.2-4.8L5 10l4.8-2.2L12 3z"></path></svg>AI Ұсыныстары</span></div>
            <div class="ai-sub">Талдау нәтижесіне сүйене отырып қысқа ұсыныс береміз</div>
            <div class="ai-list">
                <div class="ai-item">
                    <div class="ai-icon">
                        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 4c-7 0-12.5 3.7-14.6 9.4C4.9 15.2 5 18 5 20c2 0 4.8-.1 6.6-.4C16.3 17.5 20 12 20 4z"></path><path d="M7 17c2-2.5 5-4.5 9-6"></path></svg>
                    </div>
                    <div>
                        <div class="ai-item-title">Құрғақшылық қаупі</div>
                        <div class="ai-item-sub">Жоғары тәуекел үлесі {high_risk_pct:.1f}% · су үнемдеу қажет</div>
                    </div>
                </div>
                <div class="ai-item">
                    <div class="ai-icon">
                        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="6" r="2.2"></circle><circle cx="6" cy="12" r="2.2"></circle><circle cx="18" cy="12" r="2.2"></circle><circle cx="8" cy="18" r="2.2"></circle><circle cx="16" cy="18" r="2.2"></circle></svg>
                    </div>
                    <div>
                        <div class="ai-item-title">Өнімділік ұсынысы</div>
                        <div class="ai-item-sub">Орташа бағалау {avg_merit:.1f}/100 · қолдау күшейтілсін</div>
                    </div>
                </div>
            </div>
            <a class="ai-btn" href="#">Толық есепті көру</a>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# БЕТ 2: АЙМАҚТЫҚ КАРТА
# ═══════════════════════════════════════════════════════════════
elif page == "Аймақтық карта":
    st.markdown(
        """
    <div class="page-header">
        <p class="page-title">Аймақтық карта</p>
        <p class="page-subtitle">Қазақстан облыстары бойынша субсидия аналитикасы</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    engine = ScoringEngine()
    reg_df = engine.regional_report(df)
    reg_reset = reg_df.copy().reset_index()

    # Координаттар қосу
    if "lat" not in reg_reset.columns:
        reg_reset["lat"] = reg_reset["Область"].map(
            lambda r: KZ_REGIONS.get(r, {}).get("lat", 51.0)
        )
        reg_reset["lon"] = reg_reset["Область"].map(
            lambda r: KZ_REGIONS.get(r, {}).get("lon", 71.0)
        )

    reg_reset["lat"] = pd.to_numeric(reg_reset["lat"], errors="coerce").fillna(51.0)
    reg_reset["lon"] = pd.to_numeric(reg_reset["lon"], errors="coerce").fillna(71.0)

    fig_map = px.scatter_mapbox(
        reg_reset,
        lat="lat",
        lon="lon",
        size="Всего_заявок",
        color="Средний_Merit",
        hover_name="Область",
        hover_data={
            "Всего_заявок": True,
            "Процент_исполн": True,
            "Средний_Merit": ":.1f",
            "Процент_риска": True,
            "lat": False,
            "lon": False,
        },
        color_continuous_scale=["#dc2626", "#d97706", "#1a6b3c"],
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

    st.markdown(
        """
    <div class="info-box">
        Шеңбер өлшемі — өтінімдер санына сәйкес. Түсі — Merit Score деңгейін көрсетеді: жасыл — жоғары, қызыл — төмен.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">Облыстар бойынша толық кесте</p>',
        unsafe_allow_html=True,
    )

    display_cols = {
        "Всего_заявок": "Барлық өтінім",
        "Исполнено": "Орындалды",
        "Процент_исполн": "Орындалу %",
        "Средний_Merit": "Орташа Merit",
        "High_Risk": "Жоғары тәуекел",
        "Процент_риска": "Тәуекел %",
        "Общая_сумма": "Жалпы сомасы (₸)",
    }
    show_df = reg_df[[c for c in display_cols if c in reg_df.columns]].copy()
    show_df = show_df.rename(columns=display_cols)
    if "Жалпы сомасы (₸)" in show_df.columns:
        show_df["Жалпы сомасы (₸)"] = show_df["Жалпы сомасы (₸)"].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "—"
        )
    st.dataframe(style_table(show_df), use_container_width=True, height=520)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<p class="section-title">Merit Score рейтингі (ТОП-10)</p>',
            unsafe_allow_html=True,
        )
        top_m = reg_df["Средний_Merit"].dropna().sort_values(ascending=False).head(10)
        fig_m = px.bar(
            x=top_m.values,
            y=top_m.index,
            orientation="h",
            color=top_m.values,
            color_continuous_scale=["#d97706", "#1a6b3c"],
        )
        fig_m.update_layout(**make_chart_layout(360), coloraxis_showscale=False)
        fig_m.update_layout(yaxis_title="", xaxis_title="Merit Score")
        st.plotly_chart(fig_m, use_container_width=True)

    with col2:
        st.markdown(
            '<p class="section-title">Жоғары тәуекел үлесі (ТОП-10)</p>',
            unsafe_allow_html=True,
        )
        top_r = reg_df["Процент_риска"].dropna().sort_values(ascending=False).head(10)
        fig_r = px.bar(
            x=top_r.values,
            y=top_r.index,
            orientation="h",
            color=top_r.values,
            color_continuous_scale=["#1a6b3c", "#dc2626"],
        )
        fig_r.update_layout(**make_chart_layout(360), coloraxis_showscale=False)
        fig_r.update_layout(yaxis_title="", xaxis_title="Тәуекел %")
        st.plotly_chart(fig_r, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# БЕТ 3: AI СКОРИНГ
# ═══════════════════════════════════════════════════════════════
elif page == "AI Скоринг":
    st.markdown(
        """
    <div class="page-header">
        <p class="page-title">AI Скоринг жүйесі</p>
        <p class="page-subtitle">Isolation Forest · XGBoost · Merit Score алгоритмдері</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Сүзгілер
    st.markdown('<p class="section-title">Сүзгілер</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)

    with f1:
        regions_avail = ["Барлығы"] + sorted(df["Область"].unique().tolist())
        sel_region = st.selectbox("Облыс", regions_avail)
    with f2:
        dirs_avail = (
            ["Барлығы"] + sorted(df["Направление водства"].unique().tolist())
            if "Направление водства" in df.columns
            else ["Барлығы"]
        )
        sel_dir = st.selectbox("Субсидия бағыты", dirs_avail)
    with f3:
        sel_risk = st.selectbox(
            "Тәуекел деңгейі",
            ["Барлығы", "Қалыпты", "Орташа тәуекел", "Жоғары тәуекел"],
        )

    risk_map = {
        "Барлығы": None,
        "Қалыпты": "Normal",
        "Орташа тәуекел": "Medium Risk",
        "Жоғары тәуекел": "High Risk",
    }

    filtered = df.copy()
    if sel_region != "Барлығы":
        filtered = filtered[filtered["Область"] == sel_region]
    if sel_dir != "Барлығы" and "Направление водства" in filtered.columns:
        filtered = filtered[filtered["Направление водства"] == sel_dir]
    if risk_map[sel_risk] and "Risk_Level" in filtered.columns:
        filtered = filtered[filtered["Risk_Level"] == risk_map[sel_risk]]

    st.markdown(
        f"""
    <div class="info-box">
        Сүзгі нәтижесі: <strong>{len(filtered):,}</strong> өтінім табылды
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Барлық өтінім", f"{len(filtered):,}")
    k2.metric(
        "Жоғары тәуекел",
        (
            f"{(filtered['Risk_Level'] == 'High Risk').sum():,}"
            if "Risk_Level" in filtered.columns
            else "—"
        ),
    )
    k3.metric(
        "Орташа Merit",
        (
            f"{filtered['Merit_Score'].mean():.1f}"
            if "Merit_Score" in filtered.columns
            else "—"
        ),
    )
    k4.metric(
        "Орындалу ықтималы",
        f"{filtered['XGB_Prob'].mean():.1%}" if "XGB_Prob" in filtered.columns else "—",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            '<p class="section-title">Merit Score таралуы</p>', unsafe_allow_html=True
        )
        if "Merit_Score" in filtered.columns:
            fig_h = px.histogram(
                filtered, x="Merit_Score", nbins=40, color_discrete_sequence=["#1a6b3c"]
            )
            fig_h.update_layout(**make_chart_layout(280))
            fig_h.update_layout(xaxis_title="Merit Score", yaxis_title="Өтінімдер саны")
            st.plotly_chart(fig_h, use_container_width=True)

    with col_b:
        st.markdown(
            '<p class="section-title">Тәуекел деңгейлері</p>', unsafe_allow_html=True
        )
        if "Risk_Level" in filtered.columns:
            rc = filtered["Risk_Level"].value_counts().reset_index()
            rc.columns = ["Деңгей", "Саны"]
            rc["Деңгей"] = rc["Деңгей"].map(
                {"Normal": "Қалыпты", "Medium Risk": "Орташа", "High Risk": "Жоғары"}
            )
            cmap = {"Қалыпты": "#1a6b3c", "Орташа": "#d97706", "Жоғары": "#dc2626"}
            fig_rc = px.pie(
                rc,
                names="Деңгей",
                values="Саны",
                color="Деңгей",
                color_discrete_map=cmap,
                hole=0.5,
            )
            fig_rc.update_layout(**make_chart_layout(280))
            st.plotly_chart(fig_rc, use_container_width=True)

    # Үздік кандидаттар
    st.markdown(
        '<p class="section-title">Үздік кандидаттар тізімі (Шортлист)</p>',
        unsafe_allow_html=True,
    )
    engine = ScoringEngine()
    top_n = st.slider("Кандидаттар саны", 5, 100, 20)
    sl_df = engine.shortlist(
        df,
        top_n=top_n,
        region=None if sel_region == "Барлығы" else sel_region,
        direction=None if sel_dir == "Барлығы" else sel_dir,
    )

    if len(sl_df) > 0:
        st.dataframe(style_table(sl_df), use_container_width=True, height=380)
        csv_data = sl_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "CSV форматында жүктеу", csv_data, "shortlist.csv", "text/csv"
        )
    else:
        st.info("Таңдалған сүзгілер бойынша белсенді өтінімдер табылмады")

    # Жоғары тәуекел өтінімдері
    if "Risk_Level" in filtered.columns:
        high_df = filtered[filtered["Risk_Level"] == "High Risk"]
        if len(high_df) > 0:
            st.markdown(
                f'<p class="section-title">Жоғары тәуекел өтінімдері — {len(high_df):,} дана</p>',
                unsafe_allow_html=True,
            )
            cols_show = [
                c
                for c in [
                    "Номер заявки",
                    "Область",
                    "Направление водства",
                    "Причитающая сумма",
                    "Норматив",
                    "Кол_голов",
                    "Risk_Score",
                    "Risk_Reasons",
                    "Merit_Score",
                ]
                if c in high_df.columns
            ]
            st.dataframe(
                style_table(high_df[cols_show].head(200)),
                use_container_width=True,
                height=340,
            )


# ═══════════════════════════════════════════════════════════════
# БЕТ 4: БОЛЖАМДАУ
# ═══════════════════════════════════════════════════════════════
elif page == "Болжамдау":
    st.markdown(
        """
    <div class="page-header">
        <p class="page-title">Болжамдау жүйесі</p>
        <p class="page-subtitle">Exponential Smoothing алгоритмі · Уақыт сериясы талдауы</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if "Дата поступления" not in df.columns:
        st.warning("Деректерде 'Дата поступления' бағаны жоқ")
        st.stop()

    f1, f2, f3 = st.columns(3)
    with f1:
        fc_region = st.selectbox(
            "Облыс", ["Барлығы"] + sorted(df["Область"].unique().tolist())
        )
    with f2:
        fc_dir = st.selectbox(
            "Бағыт",
            (
                ["Барлығы"] + sorted(df["Направление водства"].unique().tolist())
                if "Направление водства" in df.columns
                else ["Барлығы"]
            ),
        )
    with f3:
        fc_periods = st.slider("Болжам мерзімі (ай)", 1, 6, 3)

    ts = forecasting.prepare_ts(
        df,
        region=None if fc_region == "Барлығы" else fc_region,
        direction=None if fc_dir == "Барлығы" else fc_dir,
    )

    if len(ts) < 2:
        st.warning("Болжам үшін деректер жеткіліксіз (кемінде 2 ай қажет)")
    else:
        fc_apps = forecasting.forecast_series(ts, col="Заявок", periods=fc_periods)
        hist = fc_apps[~fc_apps["is_forecast"]]
        fcast = fc_apps[fc_apps["is_forecast"]]

        st.markdown(
            '<p class="section-title">Өтінімдер саны болжамы</p>',
            unsafe_allow_html=True,
        )

        fig_fc = go.Figure()
        fig_fc.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["Заявок"],
                mode="lines+markers",
                name="Нақты деректер",
                line=dict(color="#1a6b3c", width=2.5),
                marker=dict(size=6),
            )
        )
        fig_fc.add_trace(
            go.Scatter(
                x=fcast["date"],
                y=fcast["Заявок"],
                mode="lines+markers",
                name="Болжам",
                line=dict(color="#d97706", width=2.5, dash="dash"),
                marker=dict(size=9, symbol="diamond"),
            )
        )
        fig_fc.update_layout(
            **make_chart_layout(360),
            hovermode="x unified",
        )
        fig_fc.update_layout(xaxis_title="Ай", yaxis_title="Өтінімдер саны")
        st.plotly_chart(fig_fc, use_container_width=True)

        if len(fcast) > 0:
            c1, c2, c3 = st.columns(3)
            for i, (_, row) in enumerate(fcast.iterrows()):
                [c1, c2, c3][i % 3].metric(
                    f"{row['date'].strftime('%Y-%m')}", f"{int(row['Заявок'])} өтінім"
                )

        # Сомасы болжамы
        st.markdown(
            '<p class="section-title">Субсидия сомасы болжамы</p>',
            unsafe_allow_html=True,
        )
        fc_sum = forecasting.forecast_series(ts, col="Сумма", periods=fc_periods)
        hist_s = fc_sum[~fc_sum["is_forecast"]]
        fcast_s = fc_sum[fc_sum["is_forecast"]]

        fig_sum = go.Figure()
        fig_sum.add_trace(
            go.Bar(
                x=hist_s["date"],
                y=hist_s["Сумма"] / 1e6,
                name="Нақты",
                marker_color="#1a6b3c",
                opacity=0.85,
            )
        )
        fig_sum.add_trace(
            go.Bar(
                x=fcast_s["date"],
                y=fcast_s["Сумма"] / 1e6,
                name="Болжам",
                marker_color="#d97706",
                opacity=0.75,
            )
        )
        fig_sum.update_layout(
            **make_chart_layout(300),
            barmode="overlay",
            xaxis_title="Ай",
            yaxis_title="Сомасы (млн ₸)",
        )
        st.plotly_chart(fig_sum, use_container_width=True)

    # Барлық облыстар болжамы
    st.markdown(
        '<p class="section-title">Барлық облыстар бойынша болжам кестесі</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Есептелуде..."):
        reg_fc = forecasting.regional_forecast(df, periods=3)
    if len(reg_fc) > 0:
        reg_fc.columns = [
            "Облыс",
            "Ағымдағы ай",
            "1 айдан кейін",
            "3 айдан кейін",
            "Тренд",
        ]
        st.dataframe(style_table(reg_fc), use_container_width=True, height=480)

    # Бағыттар динамикасы
    if "Направление водства" in df.columns:
        st.markdown(
            '<p class="section-title">Субсидия бағыттары динамикасы</p>',
            unsafe_allow_html=True,
        )
        dir_tr = forecasting.direction_trends(df)
        if len(dir_tr) > 0:
            fig_d = px.line(
                dir_tr,
                x="date",
                y="Заявок",
                color="Направление водства",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_d.update_layout(
                **make_chart_layout(360),
                xaxis_title="Ай",
                yaxis_title="Өтінімдер саны",
                legend_title="Бағыт",
            )
            st.plotly_chart(fig_d, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# БЕТ 5: ЖЕР ТАЛДАУЫ
# ═══════════════════════════════════════════════════════════════
elif page == "Жер талдауы":
    st.markdown(
        """
    <div class="section-head">
        <div>
            <span class="soil-chip">Интеллектуалды талдау</span>
            <p class="title" style="margin-top:8px;">Жер талдауы</p>
            <p class="sub">AI технологиялары арқылы топырақ құрамын тереңдетілген талдау</p>
        </div>
        <div class="ghost-filters">
            <span class="f">Тарих</span>
            <span class="f">Жаңа талдау</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div class="surface-panel" style="border-left:4px solid #0f8a45; margin-bottom:14px;">
        <div style="display:flex; gap:14px; align-items:flex-start;">
            <div style="background:#eaf8ef; color:#0b6d36; width:52px; height:52px; border-radius:14px; display:grid; place-items:center;">{icon_svg("eco")}</div>
            <div>
                <div style="font-size:22px; font-weight:800; color:#0f172a; line-height:1.15; margin-bottom:6px;">Жер талдауы дегеніміз не?</div>
                <div style="font-size:14px; color:#475569; line-height:1.6; max-width:980px;">Топырақтың сапасын фотосурет арқылы бағалап, ылғалдылық, құнарлылық және ықтимал агрономиялық тәуекелдерді анықтаймыз. Бұл фермерге дұрыс дақылды таңдауға, тыңайтқышты тиімді пайдалануға және шығынды азайтуға көмектеседі.</div>
                <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:12px;">
                    <span class="soil-chip">Шығымдылықты арттыру</span>
                    <span class="soil-chip">Ресурстарды үнемдеу</span>
                    <span class="soil-chip">Экологиялық тұрақтылық</span>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if "soil_history" not in st.session_state:
        st.session_state["soil_history"] = []

    result = None

    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
        st.markdown('<div class="soil-card">', unsafe_allow_html=True)
        st.markdown("<h3>Топырақ фотосын жүктеу</h3>", unsafe_allow_html=True)
        st.markdown(
            '<p class="soil-muted">JPG/PNG форматында 2–5 сурет жүктеңіз. Жақсы жарық пен жақын қашықтық дәлдікті арттырады.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="soil-upload-zone"><strong>Файлды осында сүйреңіз немесе таңдаңыз</strong><br><span class="soil-muted">Макс. 15MB · Кадрда тек топырақ көрінсін</span></div>',
            unsafe_allow_html=True,
        )

        sel_region_soil = st.selectbox(
            "Өңірді таңдаңыз",
            REGION_LIST,
            index=(
                REGION_LIST.index("Северо-Казахстанская область")
                if "Северо-Казахстанская область" in REGION_LIST
                else 0
            ),
        )

        uploaded_files = st.file_uploader(
            "Суреттерді таңдаңыз (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        images = []
        if uploaded_files:
            cols = st.columns(min(len(uploaded_files), 4))
            for i, uf in enumerate(uploaded_files):
                img = Image.open(uf)
                images.append(img)
                with cols[i % 4]:
                    st.image(img, caption=f"Фото {i + 1}", use_container_width=True)

        run_btn = st.button("Топырақ талдауын бастау", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if run_btn:
            if not images:
                st.warning("Алдымен кемінде 1 фото жүктеңіз")
            else:
                with st.spinner("AI модельдер талдау жүргізуде..."):
                    result = analyze_photos(images, sel_region_soil)

                if "error" in result:
                    st.error(result["error"])
                    result = None
                else:
                    st.session_state["soil_history"] = [
                        {
                            "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                            "soil": result["soil_kz"],
                            "region": sel_region_soil,
                            "rating": result["rating"],
                        }
                    ] + st.session_state["soil_history"][:7]

        if result is not None:
            st.markdown(
                f"""
            <div class="soil-card" style="border-left:5px solid {result['verdict_color']};">
                <h3 style="color:{result['verdict_color']};">{result['verdict']}</h3>
                <p class="soil-muted">{result['verdict_ru']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(
                f'<div class="soil-kpi"><p class="k">Топырақ түрі</p><p class="v">{result["soil_kz"]}</p></div>',
                unsafe_allow_html=True,
            )
            k2.markdown(
                f'<div class="soil-kpi"><p class="k">Сенімділік</p><p class="v">{result["confidence"]:.0f}%</p></div>',
                unsafe_allow_html=True,
            )
            k3.markdown(
                f'<div class="soil-kpi"><p class="k">Сапа рейтингі</p><p class="v">{result["rating"]}/5</p></div>',
                unsafe_allow_html=True,
            )
            k4.markdown(
                f'<div class="soil-kpi"><p class="k">Талданған фото</p><p class="v">{result["n_photos"]}</p></div>',
                unsafe_allow_html=True,
            )

            cl, cr = st.columns(2)
            with cl:
                st.markdown(
                    f"""
                <div class="soil-card">
                    <h4>Топырақ сипаттамасы</h4>
                    <p class="soil-muted"><strong>{result['soil_kz']}</strong> · {result['soil_ru']}</p>
                    <p class="soil-muted">Үлгі: {result['model_used']}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                crops_html = " ".join(
                    [
                        f"<span style='background:#eefcf3;color:#0b6d36;border:1px solid #bae7ca;padding:4px 9px;border-radius:999px;font-size:12px;font-weight:600;'>{c}</span>"
                        for c in result["recommended_crops"]
                    ]
                )
                st.markdown(
                    f"""
                <div class="soil-card">
                    <h4>Ұсынылатын дақылдар</h4>
                    <p class="soil-muted">Аймақ: {sel_region_soil}</p>
                    <div style="line-height:2.1; margin-top:8px;">{crops_html}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with cr:
                if result["problems_found"]:
                    problems_html = "".join(
                        [
                            f"<div style='padding:6px 0; border-bottom:1px solid #fde7b0; font-size:13px;'>{k} — {v} рет анықталды</div>"
                            for k, v in result["problems"].items()
                        ]
                    )
                    st.markdown(
                        f"""
                    <div class="soil-card" style="border-top:3px solid #f59e0b;">
                        <h4>Анықталған мәселелер</h4>
                        {problems_html}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                recs_html = "".join(
                    [
                        f"<div style='padding:5px 0; border-bottom:1px solid #edf2ef; font-size:13px;'>{r}</div>"
                        for r in result["recommendations"]
                    ]
                )
                st.markdown(
                    f"""
                <div class="soil-card">
                    <h4>Агрономдық ұсыныстар</h4>
                    {recs_html}
                </div>
                """,
                    unsafe_allow_html=True,
                )

                consistency = result["consistency"]
                c_color = "#0f8a45" if consistency >= 70 else "#f59e0b"
                c_text = (
                    "Фотолар бір-біріне сәйкес, нәтиже сенімді."
                    if consistency >= 70
                    else "Фотолардағы жағдай әртүрлі, бөлек тексеру ұсынылады."
                )
                st.markdown(
                    f"""
                <div class="soil-card">
                    <h4>Фотолар сәйкестігі: <span style="color:{c_color};">{consistency:.0f}%</span></h4>
                    <p class="soil-muted">{c_text}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            scores_df = pd.DataFrame(
                {
                    "Топырақ түрі": list(result["all_scores"].keys()),
                    "Ықтималдық (%)": list(result["all_scores"].values()),
                }
            ).sort_values("Ықтималдық (%)", ascending=True)
            st.markdown(
                '<p class="section-title">Топырақ түрлері бойынша сәйкестік</p>',
                unsafe_allow_html=True,
            )
            fig_s = px.bar(
                scores_df,
                x="Ықтималдық (%)",
                y="Топырақ түрі",
                orientation="h",
                color="Ықтималдық (%)",
                color_continuous_scale=["#e5e7eb", "#0f8a45"],
            )
            fig_s.update_layout(**make_chart_layout(320), coloraxis_showscale=False)
            fig_s.update_layout(yaxis_title="", xaxis_title="Ықтималдық (%)")
            st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.markdown(
                '<p class="section-title">Топырақ түрлері анықтамалығы</p>',
                unsafe_allow_html=True,
            )
            cols_ref = st.columns(2)
            for i, (soil_name, info) in enumerate(SOIL_INFO.items()):
                with cols_ref[i % 2]:
                    stars = f"Рейтинг: {info['rating']}/5"
                    color = (
                        "#0f8a45"
                        if info["rating"] >= 4
                        else "#f59e0b" if info["rating"] == 3 else "#ef4444"
                    )
                    st.markdown(
                        f"""
                    <div class="soil-card" style="border-left:4px solid {color};">
                        <div style="font-weight:700; color:#111827; font-size:14px;">
                            {info.get('kz', soil_name)}
                        </div>
                        <div style="font-size:12px; color:#6b7280; margin:4px 0;">
                            {info.get('ru', '')}
                        </div>
                        <div style="font-size:13px;">{stars}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    with right_col:
        st.markdown(
            """
        <div class="ai-card">
            <div class="ai-title"><span style="display:inline-flex; align-items:center; gap:6px;"><svg viewBox="0 0 24 24" aria-hidden="true" style="width:16px;height:16px;fill:#ffffff;display:block;"><path d="M12 3l2.2 4.8L19 10l-4.8 2.2L12 17l-2.2-4.8L5 10l4.8-2.2L12 3z"></path></svg>AI Ұсыныстары</span></div>
            <div class="ai-sub">Топырақ құрамына қарай келесі қадамды ұсынамыз</div>
            <div class="ai-list">
                <div class="ai-item">
                    <div class="ai-icon">
                        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 4c-7 0-12.5 3.7-14.6 9.4C4.9 15.2 5 18 5 20c2 0 4.8-.1 6.6-.4C16.3 17.5 20 12 20 4z"></path><path d="M7 17c2-2.5 5-4.5 9-6"></path></svg>
                    </div>
                    <div>
                        <div class="ai-item-title">Фото сапасы</div>
                        <div class="ai-item-sub">Әртүрлі нүктеден түсіру дәлдікті арттырады</div>
                    </div>
                </div>
                <div class="ai-item">
                    <div class="ai-icon">
                        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="6" r="2.2"></circle><circle cx="6" cy="12" r="2.2"></circle><circle cx="18" cy="12" r="2.2"></circle><circle cx="8" cy="18" r="2.2"></circle><circle cx="16" cy="18" r="2.2"></circle></svg>
                    </div>
                    <div>
                        <div class="ai-item-title">Топырақ күтімі</div>
                        <div class="ai-item-sub">Ылғал жоғары болса аэрация ұсынылады</div>
                    </div>
                </div>
            </div>
            <a class="ai-btn" href="#">Толық есепті көру</a>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="soil-card"><h4>Соңғы талдау тарихы</h4>', unsafe_allow_html=True)

        history = st.session_state.get("soil_history", [])
        if history:
            for item in history:
                st.markdown(
                    f"""
                <div class="soil-history-item">
                    <div style="font-size:13px; font-weight:700; color:#0f172a;">{item['soil']}</div>
                    <div style="font-size:11px; color:#64748b;">{item['region']} · {item['date']} · {item['rating']}/5</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<p class="soil-muted">Әзірге талдау жүргізілген жоқ. Фото жүктеп, бірінші талдауды бастаңыз.</p>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<p class="soil-muted" style="margin-top:12px;"><strong>Талдау аймағы:</strong> {sel_region_soil}</p></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# БЕТ 6: СУБСИДИЯ ВЕРИФИКАЦИЯСЫ
# ═══════════════════════════════════════════════════════════════
elif page == "Субсидия верификациясы":
    st.markdown(
        """
    <div class="page-header">
        <p class="page-title">Субсидия тексеру</p>
        <p class="page-subtitle">Симуляциялық ML модельдермен автоматты верификация</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    oblasts_14 = [
        "Ақмола",
        "Ақтөбе",
        "Алматы",
        "Атырау",
        "Шығыс Қазақстан",
        "Жамбыл",
        "Батыс Қазақстан",
        "Қарағанды",
        "Қостанай",
        "Қызылорда",
        "Маңғыстау",
        "Павлодар",
        "Солтүстік Қазақстан",
        "Түркістан",
    ]
    crops = ["Бидай", "Арпа", "Рапс", "Күнбағыс", "Картоп", "Жүзім", "Басқа"]

    if "subsidy_lat" not in st.session_state:
        st.session_state["subsidy_lat"] = 51.16
    if "subsidy_lon" not in st.session_state:
        st.session_state["subsidy_lon"] = 71.47

    col_form, col_map = st.columns([1.1, 1.2], gap="large")

    with col_form:
        st.markdown('<p class="section-title">Өтінім формасы</p>', unsafe_allow_html=True)
        farmer_name = st.text_input("Фермер аты-жөні", value="")
        region = st.selectbox("Облыс", oblasts_14)
        crop_type = st.selectbox("Дақыл түрі", crops)
        declared_area_ha = st.number_input(
            "Аудан (га)",
            min_value=1.0,
            max_value=100000.0,
            value=100.0,
            step=1.0,
        )
        subsidy_amount = st.number_input(
            "Сұралған сома (₸)",
            min_value=0.0,
            value=5_000_000.0,
            step=100_000.0,
        )
        application_date = st.date_input("Өтінім күні")
        latitude = st.number_input(
            "Latitude",
            min_value=40.0,
            max_value=56.0,
            value=float(st.session_state["subsidy_lat"]),
            format="%.6f",
            key="subsidy_lat",
        )
        longitude = st.number_input(
            "Longitude",
            min_value=46.0,
            max_value=88.0,
            value=float(st.session_state["subsidy_lon"]),
            format="%.6f",
            key="subsidy_lon",
        )

    with col_map:
        st.markdown('<p class="section-title">Картадан таңдау (Folium)</p>', unsafe_allow_html=True)
        fmap = folium.Map(
            location=[float(latitude), float(longitude)],
            zoom_start=6,
            control_scale=True,
            tiles="OpenStreetMap",
        )
        folium.Marker(
            [float(latitude), float(longitude)],
            tooltip="Таңдалған координат",
            icon=folium.Icon(color="green", icon="ok-sign"),
        ).add_to(fmap)
        folium.LatLngPopup().add_to(fmap)

        map_state = st_folium(
            fmap,
            key="subsidy_folium_map",
            use_container_width=True,
            height=420,
        )

        selected_lat = float(st.session_state.get("subsidy_map_lat", latitude))
        selected_lon = float(st.session_state.get("subsidy_map_lon", longitude))
        last_clicked = map_state.get("last_clicked") if isinstance(map_state, dict) else None
        if last_clicked:
            selected_lat = float(last_clicked["lat"])
            selected_lon = float(last_clicked["lng"])
            st.session_state["subsidy_map_lat"] = selected_lat
            st.session_state["subsidy_map_lon"] = selected_lon
            st.success(f"Картадан таңдалды: {selected_lat:.6f}, {selected_lon:.6f}")
        else:
            st.caption("Картадан нүкте бассаңыз, координат автоматты жаңарады.")

    verify_btn = st.button("Тексеруді бастау", use_container_width=True, key="verify_subsidy_page")

    if verify_btn:
        claim = {
            "farmer_name": farmer_name,
            "region": region,
            "crop_type": crop_type,
            "declared_area_ha": float(declared_area_ha),
            "subsidy_amount": float(subsidy_amount),
            "application_date": application_date.isoformat(),
        }

        result = None
        try:
            with st.status("Субсидия верификациясы жүріп жатыр...", expanded=True) as status:
                steps = [
                    "1/7 Өтінім мәліметтерін тексеру",
                    "2/7 Координаттарды нормализациялау",
                    "3/7 Sentinel-2 дерегін симуляциялау",
                    "4/7 YOLOv8 детекциясын іске қосу",
                    "5/7 EfficientNet және Health CNN бағалауы",
                    "6/7 U-Net ауданын есептеу",
                    "7/7 Жалпы скоринг және шешім шығару",
                ]
                for idx, step in enumerate(steps, start=1):
                    status.write(step)
                    if idx == 6:
                        result = verify(claim, selected_lat, selected_lon)
                    time.sleep(0.5)
                if result is None:
                    result = verify(claim, selected_lat, selected_lon)
                status.update(label="Тексеру аяқталды", state="complete")
        except Exception as exc:
            st.error(f"Верификация қатесі: {exc}")
            st.stop()

        decision_color = {
            "РАСТАЛДЫ": "#16a34a",
            "КҮДІКТІ": "#d97706",
            "ӨТІРІК": "#dc2626",
        }.get(result["decision"], "#374151")
        details = result["details"]

        st.markdown(
            f"""
        <div style="background:{decision_color}22; border-left:6px solid {decision_color};
                    border-radius:12px; padding:24px; margin-top:8px; text-align:center;">
            <div style="font-size:34px; font-weight:800; color:{decision_color};">{result['decision']}</div>
            <div style="font-size:20px; font-weight:700; color:#0f172a; margin-top:6px;">
                Балл: {result['score']}/100
            </div>
            <div style="font-size:14px; color:#475569; margin-top:4px;">
                Қауіп деңгейі: {result['risk_level']}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Егіс анықталды", "Иә" if details["crop_detected"] else "Жоқ")
        m2.metric("Дақыл сәйкестігі", "Сәйкес" if details["crop_match"] else "Сәйкес емес")
        m3.metric("Денсаулық", f"{details['health_score']}/5 ({details['health_label']})")
        m4.metric("Аудан айырмасы", f"{details['area_difference_pct']:.2f}%")

        with st.expander("Балл бөлінісі", expanded=True):
            labels = [
                ("Егіс анықталды", "crop_detected", 40),
                ("Аудан сәйкестігі", "area_match", 30),
                ("Денсаулық", "health", 20),
                ("Дақыл сәйкестігі", "crop_type_match", 10),
            ]
            for title, key, max_points in labels:
                points = int(result["score_breakdown"].get(key, 0))
                st.write(f"**{title}: {points}/{max_points}**")
                st.progress(points / max_points)

        with st.expander("Спутниктік деректер (симуляция)"):
            st.json(result["satellite"])
