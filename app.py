"""
NSAP Eligibility Predictor — Streamlit Web App
===============================================
A full-featured UI for the NSAP multi-class scheme classification model.

Pages
-----
  1. 🏠 Home              – project overview & scheme descriptions
  2. 🔍 Predict           – single applicant form with live prediction
  3. 📂 Batch Predict     – upload CSV, predict for all rows, download results
  4. 📊 Model Insights    – show evaluation plots & metrics from reports/

Run
---
  streamlit run app.py
"""

import os
import sys
import json
import io

import streamlit as st
import pandas as pd
import numpy as np

# ── path so imports resolve from project root ─────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

MODEL_PATH   = os.path.join(APP_DIR, "models", "best_model.pkl")
METRICS_PATH = os.path.join(APP_DIR, "models", "metrics.json")
REPORT_DIR   = os.path.join(APP_DIR, "reports")

# ── Scheme metadata ───────────────────────────────────────────────────────────
SCHEME_NAMES = {0: "IGNOAPS", 1: "IGNWPS", 2: "IGNDPS", 3: "NFBS"}
SCHEME_FULL  = {
    0: "Indira Gandhi National Old Age Pension Scheme",
    1: "Indira Gandhi National Widow Pension Scheme",
    2: "Indira Gandhi National Disability Pension Scheme",
    3: "National Family Benefit Scheme",
}
SCHEME_RULES = {
    0: "Age ≥ 60 | BPL household",
    1: "Female · Widow · Age 18–79 | BPL household",
    2: "Disability ≥ 80% · Age 18–79 | BPL household",
    3: "Primary earner deceased · Age 18–64 | BPL household",
}
SCHEME_COLORS  = {0: "#3b82d4", 1: "#7c5cd8", 2: "#e05050", 3: "#22a87a"}
SCHEME_AMOUNT  = {0: "₹200–₹500/month", 1: "₹300/month", 2: "₹300/month", 3: "₹20,000 (one-time)"}
SCHEME_ICONS   = {0: "👴", 1: "👩", 2: "♿", 3: "👨‍👩‍👧"}
SCHEME_GRADIENT = {
    0: "linear-gradient(135deg,#3b82d4,#1e4fa8)",
    1: "linear-gradient(135deg,#7c5cd8,#4e2fa8)",
    2: "linear-gradient(135deg,#e05050,#a82020)",
    3: "linear-gradient(135deg,#22a87a,#0d6b4e)",
}

STATES = [
    "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Gujarat",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Odisha", "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh",
    "West Bengal", "Himachal Pradesh", "Uttarakhand", "Punjab", "Haryana",
]
DISTRICTS = {s: [f"{s[:4].strip()}_D{i}" for i in range(1, 6)] for s in STATES}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSAP Eligibility Predictor",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── App background ── */
  .stApp { background: #f0f4f8; }

  /* ── Sidebar shell ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #0f172a 40%, #162032 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
  }
  [data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

  /* ── Sidebar – all text defaults ── */
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] div { color: #cbd5e1 !important; }

  /* ── Sidebar – nav radio items ── */
  [data-testid="stSidebar"] .stRadio > div { gap: 4px !important; }
  [data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px !important; border-radius: 10px !important;
    display: flex !important; align-items: center !important;
    font-size: 14px !important; font-weight: 500 !important;
    cursor: pointer; transition: background 0.18s, color 0.18s;
    border: 1px solid transparent;
  }
  [data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.1) !important;
  }
  /* Selected radio item */
  [data-testid="stSidebar"] .stRadio label[data-checked="true"],
  [data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ label {
    background: rgba(59,130,212,0.18) !important;
    border-color: rgba(59,130,212,0.4) !important;
    color: #93c5fd !important;
  }
  /* Hide the actual radio dot */
  [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child { display: none !important; }

  /* ── Sidebar – dividers ── */
  [data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
    margin: 10px 0 !important;
  }

  /* ── Sidebar – metric overrides ── */
  [data-testid="stSidebar"] div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important; padding: 10px 14px !important;
    box-shadow: none !important;
  }
  [data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important; font-size: 1rem !important; font-weight: 700 !important;
  }
  [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #64748b !important; font-size: 11px !important;
  }

  /* ── Sidebar – success / warning alerts ── */
  [data-testid="stSidebar"] .stAlert {
    border-radius: 10px !important;
  }
  [data-testid="stSidebar"] .stAlert[data-baseweb="notification"] {
    background: rgba(34,168,122,0.12) !important;
    border: 1px solid rgba(34,168,122,0.3) !important;
  }

  /* ── Sidebar – brand block ── */
  .sb-brand {
    padding: 24px 16px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 8px;
  }
  .sb-flag {
    width: 48px; height: 32px; border-radius: 5px; overflow: hidden;
    display: inline-block; vertical-align: middle;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    margin-bottom: 12px;
  }
  .sb-title {
    font-size: 17px; font-weight: 800; color: #f1f5f9 !important;
    letter-spacing: -0.3px; line-height: 1.2; margin-bottom: 3px;
  }
  .sb-sub {
    font-size: 11px; color: #475569 !important; line-height: 1.5;
  }
  .sb-version {
    display: inline-block; margin-top: 8px;
    background: rgba(59,130,212,0.18); border: 1px solid rgba(59,130,212,0.3);
    color: #93c5fd !important; border-radius: 20px;
    padding: 2px 10px; font-size: 10px; font-weight: 600; letter-spacing: .5px;
  }

  /* ── Sidebar – nav section label ── */
  .sb-nav-label {
    font-size: 10px; font-weight: 700; color: #334155 !important;
    text-transform: uppercase; letter-spacing: 1.2px;
    padding: 6px 14px 4px; margin-top: 4px;
  }

  /* ── Sidebar – model status card ── */
  .sb-status {
    margin: 8px 0; border-radius: 10px; padding: 12px 14px;
    border: 1px solid; font-size: 12px; font-weight: 500;
  }
  .sb-status.ok  { background: rgba(34,168,122,0.1); border-color: rgba(34,168,122,0.3);
    color: #6ee7b7 !important; }
  .sb-status.warn{ background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.3);
    color: #fcd34d !important; }
  .sb-status .sb-dot {
    width: 7px; height: 7px; border-radius: 50%; display: inline-block;
    margin-right: 7px; vertical-align: middle;
  }
  .sb-status.ok   .sb-dot { background: #22a87a; box-shadow: 0 0 6px #22a87a; }
  .sb-status.warn .sb-dot { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }

  /* ── Sidebar – metric row ── */
  .sb-metrics { display: flex; flex-direction: column; gap: 6px; margin: 6px 0; }
  .sb-metric {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 9px; padding: 8px 12px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .sb-metric .sm-label { font-size: 11px; color: #64748b !important; font-weight: 500; }
  .sb-metric .sm-value { font-size: 13px; color: #38bdf8 !important; font-weight: 700; }

  /* ── Sidebar – footer ── */
  .sb-footer {
    padding: 14px 16px; border-top: 1px solid rgba(255,255,255,0.07);
    text-align: center; font-size: 10.5px; color: #334155 !important; line-height: 1.6;
  }
  .sb-footer a { color: #475569 !important; text-decoration: none; }

  /* ── Page header banner ── */
  .page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border-radius: 16px; padding: 36px 40px 32px; margin-bottom: 28px;
    color: #fff; position: relative; overflow: hidden;
  }
  .page-header::after {
    content: ""; position: absolute; top: -40px; right: -40px;
    width: 220px; height: 220px; border-radius: 50%;
    background: rgba(255,255,255,0.04);
  }
  .page-header h1 { color: #fff !important; margin: 0 0 6px; font-size: 2rem; font-weight: 700; }
  .page-header p  { color: #94a3b8; margin: 0; font-size: 1rem; }

  /* ── Stat cards ── */
  .stat-card {
    background: #fff; border-radius: 14px; padding: 22px 24px;
    border: 1px solid #e2e8f0; text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .stat-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
  .stat-card .stat-icon { font-size: 2rem; margin-bottom: 8px; }
  .stat-card .stat-label { color: #64748b; font-size: 12px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .8px; margin-bottom: 4px; }
  .stat-card .stat-value { font-size: 1.6rem; font-weight: 700; color: #0f172a; }
  .stat-card .stat-sub   { font-size: 12px; color: #94a3b8; margin-top: 4px; }

  /* ── Scheme cards ── */
  .scheme-card {
    background: #fff; border-radius: 14px; padding: 22px;
    border: 1px solid #e2e8f0; height: 100%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .scheme-card:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.1); }
  .scheme-card .scheme-icon-wrap {
    width: 52px; height: 52px; border-radius: 12px; display: flex;
    align-items: center; justify-content: center;
    font-size: 1.5rem; margin-bottom: 14px;
  }
  .scheme-card .scheme-code {
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 6px;
  }
  .scheme-card h4 { font-size: 14px; font-weight: 600; color: #0f172a; margin: 0 0 10px; }
  .scheme-card .rule-pill {
    display: inline-block; background: #f1f5f9; color: #475569;
    border-radius: 20px; padding: 4px 12px; font-size: 11.5px;
    margin-bottom: 8px; line-height: 1.5;
  }
  .scheme-card .amount-badge {
    display: inline-block; border-radius: 8px; padding: 5px 12px;
    font-size: 12px; font-weight: 600; color: #fff; margin-top: 4px;
  }

  /* ── Result box ── */
  .result-hero {
    border-radius: 18px; padding: 32px 36px; color: #fff;
    text-align: center; position: relative; overflow: hidden;
  }
  .result-hero::before {
    content: ""; position: absolute; top: -60px; right: -60px;
    width: 200px; height: 200px; border-radius: 50%;
    background: rgba(255,255,255,0.08);
  }
  .result-hero .scheme-emoji { font-size: 3rem; display: block; margin-bottom: 8px; }
  .result-hero h2 { color: #fff !important; font-size: 2rem; margin: 0 0 6px; font-weight: 700; }
  .result-hero .scheme-full { color: rgba(255,255,255,0.85); font-size: 14px; margin-bottom: 16px; }
  .result-hero .conf-badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    backdrop-filter: blur(4px); border-radius: 30px;
    padding: 8px 22px; font-size: 18px; font-weight: 700; color: #fff;
  }

  /* ── Probability bars ── */
  .prob-row { margin-bottom: 14px; }
  .prob-label { display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 5px; }
  .prob-name  { font-weight: 600; font-size: 13px; color: #1e293b; }
  .prob-pct   { font-weight: 700; font-size: 13px; }
  .prob-track { background: #f1f5f9; border-radius: 99px; height: 10px; overflow: hidden; }
  .prob-fill  { border-radius: 99px; height: 10px; transition: width .5s ease; }

  /* ── Info cards (detail section) ── */
  .detail-card {
    background: #fff; border-radius: 14px; padding: 22px 24px;
    border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .detail-card h4 { margin: 0 0 16px; font-size: 15px; font-weight: 700; color: #0f172a; }
  .detail-row { display: flex; gap: 8px; align-items: flex-start;
    padding: 8px 0; border-bottom: 1px solid #f1f5f9; }
  .detail-row:last-child { border-bottom: none; }
  .detail-key { font-size: 12px; color: #64748b; font-weight: 500;
    min-width: 130px; padding-top: 2px; }
  .detail-val { font-size: 13px; color: #0f172a; font-weight: 600; }

  /* ── Section divider ── */
  .section-title {
    font-size: 1.1rem; font-weight: 700; color: #0f172a;
    margin: 28px 0 16px; display: flex; align-items: center; gap: 8px;
  }
  .section-title::after {
    content: ""; flex: 1; height: 1px; background: #e2e8f0;
  }

  /* ── Step cards (How-to) ── */
  .step-card {
    background: #fff; border-radius: 14px; padding: 22px;
    border: 1px solid #e2e8f0; text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .step-num {
    width: 40px; height: 40px; border-radius: 50%;
    background: linear-gradient(135deg,#3b82d4,#1e4fa8);
    color: #fff; font-weight: 700; font-size: 16px;
    display: flex; align-items: center; justify-content: center; margin: 0 auto 12px;
  }
  .step-card h4 { font-size: 14px; font-weight: 700; margin: 0 0 6px; color: #0f172a; }
  .step-card p  { font-size: 13px; color: #64748b; margin: 0; }

  /* ── Tech badge ── */
  .tech-badge {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 14px 18px; text-align: center;
  }
  .tech-badge .tech-name { font-size: 13px; font-weight: 700; color: #0f172a; }
  .tech-badge .tech-role { font-size: 11px; color: #94a3b8; margin-top: 2px; }

  /* ── Form card ── */
  .form-section {
    background: #fff; border-radius: 16px; padding: 28px;
    border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 20px;
  }
  .form-section-title {
    font-size: 13px; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: .8px; margin-bottom: 16px;
    padding-bottom: 10px; border-bottom: 1px solid #f1f5f9;
  }

  /* ── Batch summary card ── */
  .batch-stat {
    background: #fff; border-radius: 12px; padding: 18px;
    border: 1px solid #e2e8f0; text-align: center;
  }
  .batch-stat .b-num  { font-size: 1.8rem; font-weight: 700; }
  .batch-stat .b-label{ font-size: 12px; color: #64748b; font-weight: 500; }

  /* ── Alert overrides ── */
  .stAlert > div { border-radius: 10px !important; }

  /* ── Metric override ── */
  div[data-testid="metric-container"] {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 8px 16px;
    font-weight: 600; font-size: 13px;
  }

  /* ── Button ── */
  .stButton>button, .stDownloadButton>button {
    border-radius: 10px !important; font-weight: 600 !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
  }
  .stButton>button:hover, .stDownloadButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.15) !important;
  }

  /* ── Expander ── */
  [data-testid="stExpander"] { border-radius: 12px !important; }

  /* ── Deploy step list ── */
  .deploy-step {
    background: #f8fafc; border-radius: 10px; padding: 14px 18px;
    margin-bottom: 10px; display: flex; align-items: flex-start; gap: 14px;
    border: 1px solid #e2e8f0;
  }
  .deploy-step .ds-num {
    min-width: 28px; height: 28px; border-radius: 50%;
    background: #0f172a; color: #fff; font-weight: 700; font-size: 13px;
    display: flex; align-items: center; justify-content: center;
  }
  .deploy-step .ds-text { font-size: 13.5px; color: #334155; line-height: 1.5; }

  /* ── Insights metric pill ── */
  .kpi-pill {
    background: #fff; border-radius: 12px; padding: 18px 20px;
    border: 1px solid #e2e8f0; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .kpi-pill .kpi-label { font-size: 11px; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: .8px; margin-bottom: 6px; }
  .kpi-pill .kpi-value { font-size: 1.5rem; font-weight: 800; color: #0f172a; }
  .kpi-pill .kpi-sub   { font-size: 11px; color: #64748b; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_predictor():
    if not os.path.exists(MODEL_PATH):
        return None
    from predict import NSAPPredictor
    return NSAPPredictor.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


# ── Inline SVG data URI for Indian flag (no external request) ─────────────────
# Pure CSS/HTML Indian flag — no image tag, works in any Streamlit version
_INDIA_FLAG_HTML = """
<div style="
  width:56px; height:37px; border-radius:4px; overflow:hidden;
  box-shadow:0 2px 8px rgba(0,0,0,0.5); display:inline-block;
  margin-bottom:12px; flex-shrink:0;
">
  <div style="height:33.3%;background:#FF9933;"></div>
  <div style="height:33.4%;background:#fff;
    display:flex;align-items:center;justify-content:center;position:relative;">
    <!-- Ashoka Chakra via SVG inline -->
    <svg viewBox="0 0 24 24" width="13" height="13"
         xmlns="http://www.w3.org/2000/svg"
         style="position:absolute;">
      <circle cx="12" cy="12" r="10" fill="none" stroke="#000080" stroke-width="1.4"/>
      <circle cx="12" cy="12" r="1.4" fill="#000080"/>
      <!-- 24 spokes at 15° intervals -->
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(0   12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(15  12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(30  12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(45  12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(60  12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(75  12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(90  12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(105 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(120 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(135 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(150 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(165 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(180 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(195 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(210 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(225 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(240 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(255 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(270 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(285 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(300 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(315 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(330 12 12)"/>
      <line x1="12" y1="2.5"  x2="12" y2="10.6" stroke="#000080" stroke-width="0.7" transform="rotate(345 12 12)"/>
    </svg>
  </div>
  <div style="height:33.3%;background:#138808;"></div>
</div>
"""

# ── Sidebar ───────────────────────────────────────────────────────────────────
predictor = load_predictor()
metrics   = load_metrics()

with st.sidebar:

    # ── Brand block ───────────────────────────────────────────────────────────
    # Use str.replace to inject flag HTML — avoids f-string parsing its CSS braces
    _brand_html = (
        '<div class="sb-brand">FLAG_PLACEHOLDER'
        '<div class="sb-title">NSAP Predictor</div>'
        '<div class="sb-sub">National Social Assistance Programme'
        '<br>AI-Powered Eligibility Classifier</div>'
        '<span class="sb-version">v1.0 · IBM Watson ML</span>'
        '</div>'
    ).replace("FLAG_PLACEHOLDER", _INDIA_FLAG_HTML)
    st.markdown(_brand_html, unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        " ",
        ["🏠  Home", "🔍  Predict", "📂  Batch Predict",
         "📊  Model Insights"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Model status ──────────────────────────────────────────────────────────
    st.markdown('<div class="sb-nav-label">Model Status</div>', unsafe_allow_html=True)
    if predictor:
        st.markdown(
            '<div class="sb-status ok">'
            '<span class="sb-dot"></span>'
            '✔ Model loaded &amp; ready'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sb-status warn">'
            '<span class="sb-dot"></span>'
            '⚠ Model not found — run <code>python main.py</code>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Metrics ───────────────────────────────────────────────────────────────
    if metrics:
        st.markdown('<div class="sb-nav-label" style="margin-top:12px">Live Metrics</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sb-metrics">
          <div class="sb-metric">
            <span class="sm-label">🏆 Best Model</span>
            <span class="sm-value">{metrics.get("model_name", "–")}</span>
          </div>
          <div class="sb-metric">
            <span class="sm-label">🎯 Accuracy</span>
            <span class="sm-value">{metrics.get("accuracy", 0):.2%}</span>
          </div>
          <div class="sb-metric">
            <span class="sm-label">📊 F1 Weighted</span>
            <span class="sm-value">{metrics.get("f1_weighted", 0):.4f}</span>
          </div>
          <div class="sb-metric">
            <span class="sm-label">📐 F1 Macro</span>
            <span class="sm-value">{metrics.get("f1_macro", 0):.4f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-footer">'
        '© 2024 NSAP AI Project<br>'
        '<a href="https://aikosh.indiaai.gov.in" target="_blank">AI Kosh Dataset ↗</a>'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    # Header — flag injected via .replace() to avoid f-string brace conflicts
    _home_header = (
        '<div class="page-header">'
        '<div style="display:flex;align-items:center;gap:16px;margin-bottom:10px">'
        'FLAG_PLACEHOLDER'
        '<div>'
        '<h1 style="margin:0;color:#fff;font-size:1.9rem;font-weight:800">'
        'NSAP Eligibility Predictor</h1>'
        '<p style="margin:4px 0 0;color:#94a3b8;font-size:1rem">'
        'AI-powered scheme classification for India\'s National Social Assistance Programme'
        '</p>'
        '</div>'
        '</div>'
        '</div>'
    ).replace("FLAG_PLACEHOLDER", _INDIA_FLAG_HTML)
    st.markdown(_home_header, unsafe_allow_html=True)

    # Dataset info banner
    st.info(
        "📌 **Dataset source**: "
        "[AI Kosh — NSAP District-wise Pension Data]"
        "(https://aikosh.indiaai.gov.in/web/datasets/details/"
        "district_wise_pension_data_under_the_national_social_assistance_programme_nsap_1.html)",
        icon="ℹ️",
    )

    # Stats row
    st.markdown('<div class="section-title">📈 At a Glance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label, val, sub in [
        (c1, "📋", "Sub-Schemes", "4", "Classified by AI"),
        (c2, "🏘️", "States Covered", "20", "Across India"),
        (c3, "👥", "Training Records", "12,000+", "Synthetic dataset"),
        (c4, "🎯", "Target Accuracy", "97%+", "With XGBoost/LightGBM"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-icon">{icon}</div>
              <div class="stat-label">{label}</div>
              <div class="stat-value">{val}</div>
              <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Scheme cards
    st.markdown('<div class="section-title">🗂️ NSAP Sub-Schemes</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (sid, sname) in enumerate(SCHEME_NAMES.items()):
        with cols[i]:
            color = SCHEME_COLORS[sid]
            icon  = SCHEME_ICONS[sid]
            st.markdown(f"""
            <div class="scheme-card">
              <div class="scheme-icon-wrap" style="background:{color}20">
                <span style="font-size:1.5rem">{icon}</span>
              </div>
              <div class="scheme-code" style="color:{color}">{sname}</div>
              <h4>{SCHEME_FULL[sid]}</h4>
              <div class="rule-pill">📋 {SCHEME_RULES[sid]}</div><br>
              <span class="amount-badge" style="background:{color}">
                💰 {SCHEME_AMOUNT[sid]}
              </span>
            </div>""", unsafe_allow_html=True)

    # How to use
    st.markdown('<div class="section-title">🚀 How to Use</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    for col, num, title, desc in [
        (sc1, "1", "Single Predict", "Fill the applicant form on the Predict page and get an instant scheme recommendation with confidence scores."),
        (sc2, "2", "Batch Predict",  "Upload a CSV file on the Batch Predict page to classify hundreds of applicants at once and download results."),
        (sc3, "3", "Model Insights", "Check the Model Insights page for evaluation plots, CV comparison table, and per-class performance metrics."),
    ]:
        with col:
            st.markdown(f"""
            <div class="step-card">
              <div class="step-num">{num}</div>
              <h4>{title}</h4>
              <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    # Tech stack
    st.markdown('<div class="section-title">🛠️ Technology Stack</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.columns(5)
    for col, name, role in [
        (t1, "scikit-learn",       "ML Pipelines"),
        (t2, "XGBoost",            "Gradient Boosting"),
        (t3, "LightGBM",           "Fast Classifier"),
        (t4, "IBM watsonx.ai",     "Cloud Deployment"),
        (t5, "Streamlit",          "This Dashboard"),
    ]:
        with col:
            st.markdown(f"""
            <div class="tech-badge">
              <div class="tech-name">{name}</div>
              <div class="tech-role">{role}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Predict":

    st.markdown("""
    <div class="page-header">
      <h1>🔍 Predict NSAP Scheme</h1>
      <p>Fill in the applicant details below to get an instant scheme recommendation</p>
    </div>
    """, unsafe_allow_html=True)

    if predictor is None:
        st.error("🚫 Model not loaded. Run `python main.py` from the project root first.")
        st.stop()

    # ── Form ─────────────────────────────────────────────────────────────────
    with st.form("predict_form"):

        # Personal Info
        st.markdown('<div class="form-section-title">👤 Personal Information</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age    = st.number_input("Age", min_value=0, max_value=120, value=65, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            state  = st.selectbox("State", STATES)
        with col3:
            district = st.selectbox("District", DISTRICTS.get(state, [f"{state[:4]}_D1"]))

        st.markdown("---")

        # Status Flags
        st.markdown('<div class="form-section-title">🏷️ Eligibility Flags</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1:
            is_widow    = st.checkbox("Widow / Widower", help="Applicable for IGNWPS")
        with f2:
            is_disabled = st.checkbox("Person with Disability", help="Applicable for IGNDPS")
        with f3:
            earner_dec  = st.checkbox("Primary Earner Deceased", help="Applicable for NFBS")

        dis_pct = st.slider(
            "Disability % (set ≥ 80 if disability is checked above)",
            0, 100, 0,
            help="Only used when 'Person with Disability' is checked. Must be ≥ 80% to qualify for IGNDPS.",
        )

        st.markdown("---")

        # Socio-economic
        st.markdown('<div class="form-section-title">💰 Socio-economic Details</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            income  = st.number_input("Annual Household Income (₹)",
                                       min_value=0, max_value=500000, value=25000, step=1000)
        with e2:
            land    = st.number_input("Land Holding (acres)",
                                       min_value=0.0, max_value=50.0, value=0.5, step=0.1, format="%.1f")
        with e3:
            hh_size = st.number_input("Household Size", min_value=1, max_value=20, value=4)
        with e4:
            edu_map = {0: "Illiterate", 1: "Primary (up to 5th)",
                       2: "Secondary (up to 10th)", 3: "Higher"}
            rac_map = {0: "BPL Card", 1: "Antyodaya (AAY)", 2: "APL Card"}
            edu = st.selectbox("Education Level", list(edu_map.values()))

        rac = st.selectbox("Ration Card Type", list(rac_map.values()))

        st.markdown("")
        submitted = st.form_submit_button(
            "🔮  Predict Scheme",
            type="primary",
            use_container_width=True,
        )

    # ── Result ───────────────────────────────────────────────────────────────
    if submitted:
        edu_rev = {v: k for k, v in edu_map.items()}
        rac_rev = {v: k for k, v in rac_map.items()}

        applicant = {
            "state"                  : state,
            "district"               : district,
            "age"                    : int(age),
            "gender"                 : 1 if gender == "Female" else 0,
            "is_widow"               : int(is_widow),
            "is_disabled"            : int(is_disabled),
            "disability_percentage"  : int(dis_pct) if is_disabled else 0,
            "bpl_card"               : 1,
            "primary_earner_deceased": int(earner_dec),
            "annual_income"          : int(income),
            "land_holding_acres"     : float(land),
            "household_size"         : int(hh_size),
            "education_level"        : edu_rev[edu],
            "ration_card_type"       : rac_rev[rac],
        }

        with st.spinner("Analysing applicant profile …"):
            result = predictor.predict_one(applicant)

        sid   = result["predicted_scheme_id"]
        sname = result["predicted_scheme_name"]
        conf  = result["confidence"]
        grad  = SCHEME_GRADIENT[sid]
        icon  = SCHEME_ICONS[sid]

        st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

        # Hero result card
        st.markdown(f"""
        <div class="result-hero" style="background:{grad}">
          <span class="scheme-emoji">{icon}</span>
          <h2>{sname}</h2>
          <div class="scheme-full">{SCHEME_FULL[sid]}</div>
          <div class="conf-badge">Confidence: {conf:.1%}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        d1, d2 = st.columns([1, 1])

        # Scheme details
        with d1:
            st.markdown(f"""
            <div class="detail-card">
              <h4>📌 Scheme Details</h4>
              <div class="detail-row">
                <span class="detail-key">Scheme Code</span>
                <span class="detail-val">{sname}</span>
              </div>
              <div class="detail-row">
                <span class="detail-key">Full Name</span>
                <span class="detail-val">{SCHEME_FULL[sid]}</span>
              </div>
              <div class="detail-row">
                <span class="detail-key">Eligibility</span>
                <span class="detail-val">{SCHEME_RULES[sid]}</span>
              </div>
              <div class="detail-row">
                <span class="detail-key">Grant Amount</span>
                <span class="detail-val">{SCHEME_AMOUNT[sid]}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        # Probability bars
        with d2:
            st.markdown('<div class="detail-card"><h4>📊 Probability Breakdown</h4>', unsafe_allow_html=True)
            if result["probabilities"]:
                bars_html = ""
                for s, p in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
                    sid_i = [k for k, v in SCHEME_NAMES.items() if v == s][0]
                    c     = SCHEME_COLORS[sid_i]
                    pct   = int(p * 100)
                    bars_html += f"""
                    <div class="prob-row">
                      <div class="prob-label">
                        <span class="prob-name">{SCHEME_ICONS[sid_i]} {s}</span>
                        <span class="prob-pct" style="color:{c}">{p:.2%}</span>
                      </div>
                      <div class="prob-track">
                        <div class="prob-fill" style="width:{pct}%;background:{c}"></div>
                      </div>
                    </div>"""
                st.markdown(bars_html + "</div>", unsafe_allow_html=True)

        # Applicant summary
        with st.expander("🧾 View submitted applicant data"):
            st.json(applicant)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📂  Batch Predict":

    st.markdown("""
    <div class="page-header">
      <h1>📂 Batch Prediction</h1>
      <p>Upload a CSV file to classify multiple applicants at once and download results</p>
    </div>
    """, unsafe_allow_html=True)

    if predictor is None:
        st.error("🚫 Model not loaded. Run `python main.py` from the project root first.")
        st.stop()

    TEMPLATE_COLS = [
        "state", "district", "age", "gender", "is_widow", "is_disabled",
        "disability_percentage", "bpl_card", "primary_earner_deceased",
        "annual_income", "land_holding_acres", "household_size",
        "education_level", "ration_card_type",
    ]
    sample_rows = [
        ["Bihar",         "Biha_D2",  68, 0, 0, 0, 0,  1, 0, 22000, 0.3, 3, 0, 0],
        ["Uttar Pradesh", "Uttar_D1", 45, 1, 1, 0, 0,  1, 1, 18000, 0.0, 5, 0, 1],
        ["Tamil Nadu",    "Tami_D3",  35, 0, 0, 1, 90, 1, 0, 15000, 0.1, 4, 1, 0],
        ["Odisha",        "Odis_D1",  30, 1, 0, 0, 0,  1, 1, 20000, 0.5, 6, 0, 0],
    ]
    template_df = pd.DataFrame(sample_rows, columns=TEMPLATE_COLS)
    csv_bytes   = template_df.to_csv(index=False).encode()

    # Instructions + template download
    ia, ib = st.columns([2, 1])
    with ia:
        st.markdown("""
        <div class="detail-card">
          <h4>📋 Instructions</h4>
          <div class="detail-row">
            <span class="detail-key">Step 1</span>
            <span class="detail-val">Download the CSV template using the button →</span>
          </div>
          <div class="detail-row">
            <span class="detail-key">Step 2</span>
            <span class="detail-val">Fill in your applicant records (one per row)</span>
          </div>
          <div class="detail-row">
            <span class="detail-key">Step 3</span>
            <span class="detail-val">Upload the filled CSV in the upload box below</span>
          </div>
          <div class="detail-row">
            <span class="detail-key">Step 4</span>
            <span class="detail-val">Download the predictions CSV</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with ib:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️  Download CSV Template",
            data=csv_bytes,
            file_name="nsap_batch_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("")
    uploaded = st.file_uploader(
        "📁  Upload applicant CSV file",
        type=["csv"],
        help="Must contain the same columns as the template",
    )

    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.success(f"✔ Loaded **{len(df_in):,}** applicant records")

            with st.expander("👁️ Preview uploaded data (first 10 rows)"):
                st.dataframe(df_in.head(10), use_container_width=True)

            missing = [c for c in TEMPLATE_COLS if c not in df_in.columns]
            if missing:
                st.error(f"❌ Missing required columns: `{missing}`")
                st.stop()

            with st.spinner(f"Predicting for {len(df_in):,} applicants …"):
                records = df_in[TEMPLATE_COLS].to_dict(orient="records")
                results = predictor.predict_batch(records)

            df_out = df_in.copy()
            df_out["predicted_scheme"]   = [r["predicted_scheme_name"] for r in results]
            df_out["confidence"]         = [r["confidence"]            for r in results]
            df_out["scheme_description"] = [SCHEME_FULL[r["predicted_scheme_id"]] for r in results]

            st.success(f"✅ Predictions complete for **{len(df_out):,}** applicants!")
            st.markdown("")

            # Summary stats
            st.markdown('<div class="section-title">📊 Summary</div>', unsafe_allow_html=True)
            vc = df_out["predicted_scheme"].value_counts()
            bc1, bc2, bc3, bc4 = st.columns(4)
            for col, sid in zip([bc1, bc2, bc3, bc4], [0, 1, 2, 3]):
                sn = SCHEME_NAMES[sid]
                cnt = int(vc.get(sn, 0))
                pct = cnt / len(df_out) * 100
                with col:
                    st.markdown(f"""
                    <div class="batch-stat">
                      <div class="b-num" style="color:{SCHEME_COLORS[sid]}">{cnt:,}</div>
                      <div style="font-size:13px;font-weight:700;color:#0f172a;margin:4px 0">{sn}</div>
                      <div class="b-label">{pct:.1f}% of total</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("")
            st.dataframe(df_out, use_container_width=True, height=360)

            out_csv = df_out.to_csv(index=False).encode()
            st.download_button(
                label="⬇️  Download Predictions CSV",
                data=out_csv,
                file_name="nsap_predictions.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Insights":

    st.markdown("""
    <div class="page-header">
      <h1>📊 Model Insights & Evaluation</h1>
      <p>Performance metrics and visual analysis of the trained classifier</p>
    </div>
    """, unsafe_allow_html=True)

    metrics = load_metrics()
    if metrics is None:
        st.warning("⚠️ No metrics found. Run `python main.py` to train the model first.")
        st.stop()

    # ── KPI Pills ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 Test Set Performance</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    for col, label, raw_val, fmt, sub in [
        (k1, "Best Model",    metrics.get("model_name", "–"),          None,      "Selected by CV F1"),
        (k2, "Accuracy",      metrics.get("accuracy", 0),              ".4f",     "On held-out test set"),
        (k3, "F1 (Weighted)", metrics.get("f1_weighted", 0),           ".4f",     "Weighted average"),
        (k4, "F1 (Macro)",    metrics.get("f1_macro", 0),              ".4f",     "Macro average"),
    ]:
        display = f"{raw_val:{fmt}}" if fmt else str(raw_val)
        with col:
            st.markdown(f"""
            <div class="kpi-pill">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{display}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    if metrics.get("roc_auc_weighted"):
        st.metric("ROC-AUC (OvR, Weighted)", f"{metrics['roc_auc_weighted']:.4f}")

    st.markdown("")

    # ── CV Table + Per-class report side-by-side ─────────────────────────────
    left, right = st.columns(2)

    with left:
        if "cv_results" in metrics:
            st.markdown('<div class="section-title">🔁 Cross-Validation Comparison</div>', unsafe_allow_html=True)
            cv_df = pd.DataFrame([
                {"Model": name, "Mean F1": v["mean_f1"], "Std F1": v["std_f1"]}
                for name, v in metrics["cv_results"].items()
            ]).sort_values("Mean F1", ascending=False).reset_index(drop=True)
            st.dataframe(
                cv_df.style
                    .format({"Mean F1": "{:.4f}", "Std F1": "{:.4f}"})
                    .background_gradient(subset=["Mean F1"], cmap="Greens")
                    .highlight_max(subset=["Mean F1"], color="#d1fae5"),
                use_container_width=True,
                hide_index=True,
            )

    with right:
        if "classification_report" in metrics:
            st.markdown('<div class="section-title">📋 Per-Class Report</div>', unsafe_allow_html=True)
            report = metrics["classification_report"]
            rows = []
            for cls_name in [SCHEME_NAMES[i] for i in range(4)]:
                if cls_name in report:
                    r = report[cls_name]
                    rows.append({
                        "Scheme"   : cls_name,
                        "Precision": round(r["precision"], 4),
                        "Recall"   : round(r["recall"],    4),
                        "F1-Score" : round(r["f1-score"],  4),
                        "Support"  : int(r["support"]),
                    })
            rep_df = pd.DataFrame(rows)
            st.dataframe(
                rep_df.style
                    .format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"})
                    .background_gradient(subset=["F1-Score"], cmap="Blues"),
                use_container_width=True,
                hide_index=True,
            )

    # ── Evaluation Plots ─────────────────────────────────────────────────────
    PLOTS = {
        "Class Distribution"      : "class_distribution.png",
        "Age Distribution"        : "age_by_scheme.png",
        "Correlation Heatmap"     : "correlation_heatmap.png",
        "Model CV Comparison"     : "cv_comparison.png",
        "Confusion Matrix"        : "confusion_matrix.png",
        "Per-Class F1 Scores"     : "per_class_f1.png",
        "Feature Importance"      : "feature_importance.png",
    }
    available = {k: v for k, v in PLOTS.items()
                 if os.path.exists(os.path.join(REPORT_DIR, v))}

    if available:
        st.markdown('<div class="section-title">📈 Evaluation Plots</div>', unsafe_allow_html=True)
        tabs = st.tabs(list(available.keys()))
        for tab, (title, fname) in zip(tabs, available.items()):
            with tab:
                st.image(os.path.join(REPORT_DIR, fname), use_container_width=True)
    else:
        st.info("📂 No plots found yet. Run `python main.py` to generate them into `reports/`.")

    # Fallback confusion matrix from JSON
    if "confusion_matrix" in metrics and not available:
        st.markdown('<div class="section-title">🔲 Confusion Matrix (raw)</div>', unsafe_allow_html=True)
        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=[SCHEME_NAMES[i] for i in range(4)],
            columns=[SCHEME_NAMES[i] for i in range(4)],
        )
        st.dataframe(cm, use_container_width=True)


