"""
Port Intelligence AI — Application Streamlit refactorisée
Master 1 IA — Université de Djibouti — 2026
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
import base64
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Dict, Any

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Port Intelligence AI — DMP",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Constants
VALID_PAGES = ["home", "predict", "compare", "about", "analytics"]
MODELES_DATA = [
    ("Régression Linéaire", 0.157, 480, 780,  "#f43f5e"),
    ("KNN",                 0.587, 246, 546,  "#f97316"),
    ("Arbre de Décision",   0.490, 353, 607,  "#eab308"),
    ("Forêt Aléatoire ★",   0.667, 260, 552,  "#22c55e"),
]

DATA_HEURES = [
    ("16h", 2700), ("17h", 2500), ("15h", 2000), ("23h", 2000),
    ("11h", 1800), ("19h", 1600), ("1h",  1600), ("20h", 1400),
    ("18h", 1300), ("12h", 1300), ("22h", 1700), ("13h", 1200),
]

DATA_CARGO = [
    ("Steel",       4500, "#f43f5e"),
    ("I-beam",      2600, "#f97316"),
    ("*Rebar",      1500, "#eab308"),
    ("Wheat",       1500, "#eab308"),
    ("Rice",        1300, "#22c55e"),
    ("General",     800,  "#22c55e"),
    ("Fertilizer",  800,  "#22c55e"),
    ("Vehicle",     700,  "#22c55e"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def img_to_b64(path: str) -> str | None:
    """Convertir une image en base64."""
    if not os.path.exists(path):
        return None
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp"}.get(ext, "image/png")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

@st.cache_resource
def load_model():
    """Charger le modèle ML."""
    path = "meilleur_modele_foret_simple.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def get_model_metadata() -> Tuple[list, list, list, str, str]:
    """Extraire les métadonnées du modèle."""
    pipe = load_model()
    if pipe is not None:
        cat_step = pipe.named_steps['comp'].named_transformers_['cat']
        ohe = cat_step.named_steps['onehot']
        imp_cat = cat_step.named_steps['imputer']
        return (
            sorted(ohe.categories_[0].tolist()),
            sorted(ohe.categories_[1].tolist()),
            sorted(ohe.categories_[4].tolist()),
            imp_cat.statistics_[2],
            imp_cat.statistics_[3]
        )
    return (
        ['Fertilizer','Steel Bar','Steel Coil','Vehicle','Wheat','Cement','Sugar','Rice','Clinker','General Cargo'],
        ['ABBAY II','GALINI','AFEA'],
        ['Delivery','Direct Discharge','Receiving'],
        "HASSAN.MAHAMOUD",
        "HASSAN.MAHAMOUD"
    )

def apply_css_styles():
    """Appliquer les styles CSS globaux."""
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
    :root {
      --accent: #1d9e75;
      --accent-2: #0a2540;
      --accent-glow: rgba(29,158,117,0.2);
      --bg: #06090f;
      --bg-2: #0a0f1a;
      --surface: rgba(255,255,255,0.04);
      --surface-h: rgba(255,255,255,0.07);
      --border: rgba(255,255,255,0.08);
      --border-h: rgba(29,158,117,0.4);
      --text: #ffffff;
      --text-muted: rgba(255,255,255,0.5);
      --text-dim: rgba(255,255,255,0.25);
      --r-sm: 10px; --r-md: 16px; --r-lg: 24px;
      --tr: 0.32s cubic-bezier(0.22,1,0.36,1);
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background: #06090f; min-height: 100vh; }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0a0f1a; }
    ::-webkit-scrollbar-thumb { background: rgba(29,158,117,0.4); border-radius: 99px; }
    [data-testid="stSidebar"] { display: none; }

    /* ══ NAVBAR ══ */
    .navbar a, .navbar a:hover, .navbar a:visited, .navbar a:focus, .navbar a:active { text-decoration: none !important; }
    .navbar {
        position: fixed; top:0; left:0; right:0; z-index:9999;
        display:flex; align-items:center; justify-content:space-between;
        padding:0 2.5rem; height:64px;
        background:rgba(6,9,15,0.88);
        backdrop-filter:blur(20px);
        border-bottom:1px solid rgba(255,255,255,0.07);
    }
    .navbar-logo { display:flex; align-items:center; gap:10px; cursor:pointer; }
    .navbar-logo-icon { width:36px; height:36px; background:linear-gradient(135deg,#1d9e75,#0a2540); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1rem; color:white; overflow:hidden; }
    .navbar-logo-icon img { width:100%; height:100%; object-fit:contain; }
    .navbar-logo-text { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; color:#fff; letter-spacing:0.02em; }
    .navbar-logo-text span { color:#1d9e75; }
    .navbar-links { display:flex; align-items:center; gap:0.3rem; }
    .nav-btn { display:inline-flex; align-items:center; gap:8px; padding:7px 16px; border-radius:8px; font-size:0.85rem; font-weight:500; color:rgba(255,255,255,0.55); border:none; background:transparent; transition:all 0.2s; white-space:nowrap; text-decoration:none; }
    .nav-btn:hover, .nav-btn.active { color:#fff; background:rgba(255,255,255,0.07); }
    .nav-btn.active { color:#4ade80; }
    .nav-btn i { font-size:0.78rem; }

    /* ══ PAGE WRAPPER ══ */
    .page-wrapper { padding-top:80px; max-width:1200px; margin:0 auto; padding-left:2rem; padding-right:2rem; }

    /* ══ HERO ══ */
    .hero-section { padding:5rem 0 1.5rem; text-align:center; }
    .hero-pill { display:inline-flex; align-items:center; gap:8px; background:rgba(29,158,117,0.12); border:1px solid rgba(29,158,117,0.3); border-radius:99px; padding:5px 16px; font-size:0.78rem; font-weight:600; color:#4ade80; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:1.5rem; }
    .hero-title { font-family:'Syne',sans-serif; font-size:clamp(3.2rem,8vw,6rem); font-weight:800; color:#fff; line-height:1.05; margin:0 0 0.8rem; }
    .hero-title .gradient-text { background:linear-gradient(135deg,#1d9e75 0%,#22d3ee 50%,#3b82f6 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
    .hero-subtitle-inline { font-size:1rem; color:rgba(255,255,255,0.45); max-width:620px; margin:0 auto 1.8rem; line-height:1.7; font-weight:300; }
    .stats-band { display:grid; grid-template-columns:repeat(4,1fr); gap:1.2rem; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:24px; padding:2.8rem 2rem; margin-bottom:4rem; text-align:center; }
    .stat-num { font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800; background:linear-gradient(135deg,#1d9e75,#22d3ee); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
    .stat-lbl { font-size:0.88rem; color:rgba(255,255,255,0.4); margin-top:0.4rem; }

    /* ══ PAGE HEADER ══ */
    .page-header { padding:3rem 0 2rem; }
    .page-header-eyebrow { font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.12em; color:#1d9e75; margin-bottom:0.6rem; display:flex; align-items:center; gap:8px; }
    .page-header-title { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:white; margin:0 0 0.6rem; }
    .page-header-sub { font-size:0.95rem; color:rgba(255,255,255,0.45); }

    /* ══ GLASS CARD ══ */
    .glass-card { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:20px; padding:1.5rem; margin-bottom:1rem; }
    .glass-card-title { font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:rgba(255,255,255,0.3); margin-bottom:1rem; display:flex; align-items:center; gap:8px; }

    /* ══ RESULT HERO ══ */
    .result-hero { background:linear-gradient(135deg,rgba(29,158,117,0.12),rgba(59,130,246,0.12)); border:1px solid rgba(29,158,117,0.25); border-radius:20px; padding:2.5rem 1.5rem; text-align:center; margin-bottom:1rem; position:relative; overflow:hidden; }
    .result-hero::after { content:''; position:absolute; top:-60px; right:-60px; width:180px; height:180px; border-radius:50%; background:radial-gradient(circle,rgba(29,158,117,0.12),transparent 70%); pointer-events:none; }
    .result-time { font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; color:white; margin:0; }
    .result-hmin { font-size:1.1rem; font-weight:500; background:linear-gradient(135deg,#1d9e75,#22d3ee); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin:0.3rem 0 0; }
    .result-sub { font-size:0.85rem; color:rgba(255,255,255,0.4); }

    /* ══ GAUGE ══ */
    .gauge-wrap { padding:1.2rem 0 0.6rem; }
    .gauge-bar-bg { height:14px; border-radius:99px; background:linear-gradient(90deg,#16a34a 0%,#16a34a 40%,#d97706 40%,#d97706 70%,#dc2626 70%,#dc2626 100%); position:relative; margin-bottom:6px; }
    .gauge-needle { position:absolute; top:-5px; width:4px; height:24px; background:white; border-radius:2px; transform:translateX(-50%); box-shadow:0 0 8px rgba(255,255,255,0.4); transition:left 0.8s cubic-bezier(.34,1.56,.64,1); }
    .gauge-labels { display:flex; justify-content:space-between; font-size:0.68rem; color:rgba(255,255,255,0.3); }
    .risk-badge-wrap { text-align:center; margin-top:12px; }
    .risk-badge { display:inline-block; font-size:0.9rem; font-weight:700; padding:7px 22px; border-radius:99px; }
    .risk-low    { background:rgba(22,163,74,0.2);  border:1px solid rgba(22,163,74,0.4);  color:#4ade80; }
    .risk-medium { background:rgba(217,119,6,0.2);  border:1px solid rgba(217,119,6,0.4);  color:#fbbf24; }
    .risk-high   { background:rgba(220,38,38,0.2);  border:1px solid rgba(220,38,38,0.4);  color:#f87171; }

    /* ══ FACTEURS ══ */
    .facteur-row { display:flex; align-items:center; gap:10px; padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.06); }
    .facteur-row:last-child { border-bottom:none; }
    .facteur-icon { font-size:0.8rem; width:18px; text-align:center; }
    .facteur-label { flex:1; font-size:0.83rem; font-weight:500; color:rgba(255,255,255,0.7); }
    .facteur-val { font-size:0.83rem; color:rgba(255,255,255,0.4); }

    /* ══ TABLE ══ */
    .hist-table { width:100%; border-collapse:collapse; font-size:0.82rem; margin-top:8px; }
    .hist-table th { background:rgba(255,255,255,0.05); color:rgba(255,255,255,0.3); font-weight:600; padding:8px 10px; text-align:left; border-bottom:1px solid rgba(255,255,255,0.07); font-size:0.68rem; text-transform:uppercase; letter-spacing:0.05em; }
    .hist-table td { padding:7px 10px; border-bottom:1px solid rgba(255,255,255,0.04); color:rgba(255,255,255,0.7); }
    .hist-table tr:last-child td { border-bottom:none; }
    .tag-low    { background:rgba(22,163,74,0.15);  color:#4ade80;  padding:2px 9px; border-radius:99px; font-size:0.72rem; font-weight:600; }
    .tag-medium { background:rgba(217,119,6,0.15);  color:#fbbf24;  padding:2px 9px; border-radius:99px; font-size:0.72rem; font-weight:600; }
    .tag-high   { background:rgba(220,38,38,0.15);  color:#f87171;  padding:2px 9px; border-radius:99px; font-size:0.72rem; font-weight:600; }

    /* ══ PIPELINE ══ */
    .pipeline-container { display:flex; align-items:center; justify-content:space-around; padding:3rem 2rem; background:rgba(255,255,255,0.03); border-radius:24px; margin:2rem 0; border:1px solid rgba(255,255,255,0.07); }
    .pipeline-step-box { display:flex; flex-direction:column; align-items:center; gap:12px; flex:1; text-align:center; position:relative; }
    .pipeline-step-box::after { content:''; position:absolute; right:-40px; top:30px; width:80px; height:3px; background:linear-gradient(90deg,#1d9e75,transparent); }
    .pipeline-step-box:last-child::after { display:none; }
    .step-icon { width:60px; height:60px; border-radius:50%; background:linear-gradient(135deg,#1d9e75,#0a7a5a); display:flex; align-items:center; justify-content:center; font-size:1.8rem; color:white; }
    .step-label { font-size:0.85rem; font-weight:600; color:rgba(255,255,255,0.8); }
    .step-desc { font-size:0.7rem; color:rgba(255,255,255,0.4); }

    /* ══ FOOTER ══ */
    .site-footer { border-top:1px solid rgba(255,255,255,0.07); padding:2.8rem 0 1.8rem; margin-top:4rem; }
    .site-footer a, .site-footer a:hover, .site-footer a:visited, .site-footer a:focus, .site-footer a:active { text-decoration:none !important; }
    .footer-grid { display:grid; grid-template-columns:1.4fr 1fr 1fr; gap:3rem; margin-bottom:2.2rem; }
    .footer-brand-row { display:flex; align-items:center; gap:10px; margin-bottom:0.9rem; }
    .footer-logo-box { width:36px; height:36px; background:linear-gradient(135deg,#1d9e75,#0a2540); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:14px; color:white; overflow:hidden; flex-shrink:0; }
    .footer-app-name { font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; color:white; }
    .footer-app-name span { color:#1d9e75; }
    .footer-desc { font-size:0.82rem; color:rgba(255,255,255,0.38); line-height:1.7; font-weight:300; max-width:240px; }
    .footer-col-title { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:rgba(255,255,255,0.22); margin-bottom:0.9rem; }
    .footer-link { display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:8px; font-size:0.82rem; font-weight:500; color:rgba(255,255,255,0.42); text-decoration:none !important; margin-bottom:0.2rem; transition:color 0.2s, background 0.2s; width:100%; }
    .footer-link i { font-size:0.72rem; flex-shrink:0; }
    .footer-link:hover { color:#fff; background:rgba(255,255,255,0.07); }
    .footer-bottom { padding-top:1.4rem; border-top:1px solid rgba(255,255,255,0.07); display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.8rem; }
    .footer-copy { font-size:0.74rem; color:rgba(255,255,255,0.22); }
    .footer-copy span { color:#1d9e75; }

    /* ══ Streamlit overrides ══ */
    .stApp > div { background: transparent !important; }
    .stSelectbox > div > div { background:rgba(255,255,255,0.05) !important; border:1px solid rgba(255,255,255,0.1) !important; border-radius:10px !important; color:white !important; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color:rgba(255,255,255,0.65) !important; font-size:0.85rem !important; }
    .stSlider > div > div > div { background:linear-gradient(135deg,#1d9e75,#22d3ee) !important; }
    .stNumberInput > div > div > input { background:rgba(255,255,255,0.05) !important; border:1px solid rgba(255,255,255,0.1) !important; border-radius:10px !important; color:white !important; }
    div[data-testid="stButton"] > button { background:linear-gradient(135deg,#1d9e75,#0f6e56) !important; color:white !important; border:none !important; border-radius:12px !important; font-size:0.95rem !important; font-weight:600 !important; padding:12px 28px !important; width:100% !important; }
    div[data-testid="stButton"] > button:hover { opacity:0.9 !important; }
    hr { border-color:rgba(255,255,255,0.07) !important; }
    div[data-testid="column"] { padding:0 0.4rem !important; }

    @media(max-width:900px){
        .stats-band { grid-template-columns:repeat(2,1fr); }
        .pipeline-container { flex-direction:column; }
        .pipeline-step-box::after { right:0; top:60px; width:3px; height:30px; }
    }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE & ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

if "page" not in st.session_state:
    st.session_state.page = "home"
if "historique" not in st.session_state:
    st.session_state.historique = []

if "page" in st.query_params:
    p = st.query_params["page"]
    if p in VALID_PAGES:
        st.session_state.page = p

# ═══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_navbar():
    """Afficher la barre de navigation."""
    pages_nav = [
        ("home",      "fa-anchor",      "Accueil"),
        ("predict",   "fa-clock",       "Prédiction"),
        ("compare",   "fa-chart-bar",   "Comparaison"),
        ("analytics", "fa-chart-line",  "Analytics"),
        ("about",     "fa-circle-info", "À propos"),
    ]

    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        uri = img_to_b64(logo_path)
        logo_html = f'<img src="{uri}" style="width:36px;height:36px;border-radius:10px;object-fit:contain;" />'
    else:
        logo_html = '<i class="fa-solid fa-anchor"></i>'

    nav_links = ""
    for pid, icon, label in pages_nav:
        active_cls = "active" if st.session_state.page == pid else ""
        nav_links += f'<a class="nav-btn {active_cls}" href="?page={pid}" target="_self"><i class="fa-solid {icon}"></i> {label}</a>'

    st.markdown(f"""
    <div class="navbar">
      <div class="navbar-logo">
        <div class="navbar-logo-icon">{logo_html}</div>
        <span class="navbar-logo-text">Port<span>Intelligence</span></span>
      </div>
      <div class="navbar-links">{nav_links}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE — PIPELINE ANIMÉ
# ═══════════════════════════════════════════════════════════════════════════════

def render_home_page():
    """Page d'accueil avec pipeline animé."""
    st.markdown("""
    <div class="hero-section">
        <div class="hero-pill"><i class="fa-solid fa-ship"></i> Machine Learning · Doraleh Port · Djibouti</div>
        <h1 class="hero-title">
            Prédire la surestarie<br>avec l'<span class="gradient-text">Intelligence Artificielle</span>
        </h1>
        <p class="hero-subtitle-inline">
            Modèle Forêt Aléatoire entraîné sur 62 329 opérations réelles du port DMP.
            Prédisez la durée d'attente des camions dès leur entrée au terminal.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline animé
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step-box">
            <div class="step-icon"><i class="fa-solid fa-gate"></i></div>
            <div class="step-label">Gate In</div>
            <div class="step-desc">Entrée du camion</div>
        </div>
        <div class="pipeline-step-box">
            <div class="step-icon"><i class="fa-solid fa-weight-scale"></i></div>
            <div class="step-label">Pesée 1</div>
            <div class="step-desc">Poids à vide</div>
        </div>
        <div class="pipeline-step-box">
            <div class="step-icon"><i class="fa-solid fa-box"></i></div>
            <div class="step-label">Chargement</div>
            <div class="step-desc">Cargaison</div>
        </div>
        <div class="pipeline-step-box">
            <div class="step-icon"><i class="fa-solid fa-weight-scale"></i></div>
            <div class="step-label">Pesée 2</div>
            <div class="step-desc">Poids total</div>
        </div>
        <div class="pipeline-step-box">
            <div class="step-icon"><i class="fa-solid fa-arrow-right-from-bracket"></i></div>
            <div class="step-label">Gate Out</div>
            <div class="step-desc">Sortie</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-band">
        <div><div class="stat-num">62K</div><div class="stat-lbl">Enregistrements réels</div></div>
        <div><div class="stat-num">0.667</div><div class="stat-lbl">R² — Forêt Aléatoire</div></div>
        <div><div class="stat-num">260</div><div class="stat-lbl">MAE (minutes)</div></div>
        <div><div class="stat-num">4</div><div class="stat-lbl">Algorithmes comparés</div></div>
    </div>

    <div class="glass-card">
        <div class="glass-card-title"><i class="fa-solid fa-lightbulb"></i> Comment ça marche?</div>
        <p style="font-size:0.9rem;color:rgba(255,255,255,0.6);line-height:1.8;margin:0;">
            <strong>Port Intelligence</strong> utilise un modèle <strong>Forêt Aléatoire</strong> pour estimer la durée d'attente d'un camion au terminal.
            En analysant 62 329 opérations réelles, le modèle apprend les patterns de surestarie basés sur:
        </p>
        <ul style="font-size:0.88rem;color:rgba(255,255,255,0.55);margin-top:1rem;">
            <li>🕐 Heure d'entrée (pics d'activité détectés)</li>
            <li>📦 Type et poids de la cargaison</li>
            <li>🚢 Navire et opérateur</li>
            <li>📅 Jour et mois de l'opération</li>
        </ul>
        <p style="font-size:0.85rem;color:rgba(255,255,255,0.4);margin-top:1rem;margin-bottom:0;">
            ✨ Résultat: <strong>MAE = 260 minutes</strong> — une prédiction fiable pour optimiser la logistique portuaire.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def render_predict_page():
    """Page de prédiction."""
    NOM_CARGAISON, NOM_NAVIRE, TYPE_TRAVAIL, OP_ENTREE_DEFAULT, OP_SORTIE_DEFAULT = get_model_metadata()

    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow"><i class="fa-solid fa-clock"></i> Prédiction en temps réel</div>
        <h1 class="page-header-title">Estimer la surestarie</h1>
        <p class="page-header-sub">Renseignez les informations du camion à son entrée Gate In pour obtenir une estimation instantanée.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-band" style="margin-bottom:2rem;padding:1.6rem 1.5rem;">
        <div><div class="stat-num" style="font-size:1.8rem;">768</div><div class="stat-lbl">Surestarie moy. (min)</div></div>
        <div><div class="stat-num" style="font-size:1.8rem;">31 873</div><div class="stat-lbl">Camions traités</div></div>
        <div><div class="stat-num" style="font-size:1.8rem;color:#f87171;">11,17%</div><div class="stat-lbl">Taux High Demurrage</div></div>
        <div><div class="stat-num" style="font-size:1.8rem;">16h</div><div class="stat-lbl">Heure de pointe</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="glass-card"><div class="glass-card-title"><i class="fa-solid fa-weight-hanging"></i> Poids (kg)</div>', unsafe_allow_html=True)
        poids_tare = st.number_input("Poids Tare (kg)", value=8500, min_value=0, max_value=50000, step=500, help="Poids du camion à vide")
        poids_cargaison = st.number_input("Poids Cargaison (kg)", value=15000, min_value=0, max_value=50000, step=500, help="Poids de la cargaison")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card"><div class="glass-card-title"><i class="fa-solid fa-calendar"></i> Temporel</div>', unsafe_allow_html=True)
        heure = st.slider("Heure d'entrée (0–23)", 0, 23, 14, help="Heure d'entrée au Gate In")
        jour = st.slider("Jour (0=Lundi · 6=Dimanche)", 0, 6, 1, help="Jour de la semaine")
        mois = st.slider("Mois", 1, 12, 3, help="Mois de l'opération")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="glass-card"><div class="glass-card-title"><i class="fa-solid fa-ship"></i> Opération</div>', unsafe_allow_html=True)
        nom_carg = st.selectbox("Type de cargaison", NOM_CARGAISON)
        nom_nav = st.selectbox("Navire", NOM_NAVIRE)
        type_trav = st.selectbox("Type de travail", TYPE_TRAVAIL)
        st.markdown('</div>', unsafe_allow_html=True)

    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        col_pred, col_reset = st.columns(2)
        with col_pred:
            predict_btn = st.button("🔮 Prédire", use_container_width=True)
        with col_reset:
            reset_btn = st.button("🔄 Réinitialiser", use_container_width=True)

    if reset_btn:
        st.session_state.historique = []
        st.rerun()

    if predict_btn:
        pipe = load_model()
        if pipe is None:
            st.error("⚠️ Modèle non chargé. Vérifiez que meilleur_modele_foret_simple.pkl est présent.")
        else:
            poids_entree = poids_tare
            data = pd.DataFrame([{
                'Poids_Tare_kg': poids_tare,
                'Poids_Cargaison_kg': poids_cargaison,
                'Poids_Camion_Entree_kg': poids_entree,
                'heure_entree': heure,
                'jour_semaine': jour,
                'mois': mois,
                'log_Poids_Tare_kg': np.log1p(poids_tare),
                'log_Poids_Cargaison_kg': np.log1p(poids_cargaison),
                'log_Poids_Camion_Entree_kg': np.log1p(poids_entree),
                'Nom_Cargaison': nom_carg,
                'Nom_Navire': nom_nav,
                'Operateur_Entree': OP_ENTREE_DEFAULT,
                'Operateur_Sortie': OP_SORTIE_DEFAULT,
                'Type_Travail': type_trav,
            }])
            
            pred = pipe.predict(data)[0]
            heures = int(pred // 60)
            mins = int(pred % 60)
            risque_txt = "Élevé" if pred > 1440 else "Modéré" if pred > 600 else "Faible"
            risk_cls = "risk-high" if pred > 1440 else "risk-medium" if pred > 600 else "risk-low"
            risk_icon = "fa-triangle-exclamation" if pred > 1440 else "fa-circle-exclamation" if pred > 600 else "fa-circle-check"
            needle_pct = min(pred / 2880 * 100, 100)
            conseil = "🚨 Surestarie critique — alerte immédiate recommandée au client." if pred > 1440 else \
                      "⚠️ Attente significative — informer le client importateur." if pred > 600 else \
                      "✅ Opération standard — durée d'attente normale."

            st.session_state.historique.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "cargo": nom_carg,
                "heure": heure,
                "poids": poids_cargaison,
                "pred": pred,
                "risque": risque_txt
            })

            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown(f"""
                <div class="result-hero">
                  <div class="result-time">{pred:.0f} min</div>
                  <div class="result-hmin">{heures}h {mins:02d}min</div>
                  <div class="result-sub">durée d'attente estimée au terminal</div>
                  <div class="gauge-wrap">
                    <div class="gauge-bar-bg">
                      <div class="gauge-needle" style="left:{needle_pct:.1f}%"></div>
                    </div>
                    <div class="gauge-labels"><span>0</span><span>Faible</span><span>Modéré</span><span>Élevé</span><span>2880 min</span></div>
                  </div>
                  <div class="risk-badge-wrap">
                    <span class="risk-badge {risk_cls}"><i class="fa-solid {risk_icon}"></i> Risque {risque_txt}</span>
                  </div>
                  <div style="font-size:0.82rem;color:rgba(255,255,255,0.4);margin-top:10px;">{conseil}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                alerte_h = "⚠" if 15 <= heure <= 17 else "✓"
                alerte_c = "#f87171" if 15 <= heure <= 17 else "#4ade80"
                alerte_p = "⚠" if poids_cargaison > 40000 else "✓"
                color_p = "#f87171" if poids_cargaison > 40000 else "#4ade80"
                alerte_cargo = "⚠" if nom_carg in ["Steel Bar", "Steel Coil", "Wheat", "Rice"] else "✓"
                color_cargo = "#f87171" if nom_carg in ["Steel Bar", "Steel Coil", "Wheat", "Rice"] else "#4ade80"
                vs_moy = pred - 768
                vs_sign = "+" if vs_moy >= 0 else ""
                vs_col = "#f87171" if vs_moy > 0 else "#4ade80"

                st.markdown(f"""
                <div class="glass-card">
                  <div class="glass-card-title"><i class="fa-solid fa-magnifying-glass-chart"></i> Analyse de l'opération</div>
                  <div class="facteur-row">
                    <span class="facteur-icon" style="color:{alerte_c}">{alerte_h}</span>
                    <span class="facteur-label">Heure d'entrée</span>
                    <span class="facteur-val">{heure}h {'— pic 15-17h' if 15<=heure<=17 else '— hors pic'}</span>
                  </div>
                  <div class="facteur-row">
                    <span class="facteur-icon" style="color:{color_cargo}">{alerte_cargo}</span>
                    <span class="facteur-label">Type de cargaison</span>
                    <span class="facteur-val">{nom_carg}</span>
                  </div>
                  <div class="facteur-row">
                    <span class="facteur-icon" style="color:{color_p}">{alerte_p}</span>
                    <span class="facteur-label">Poids cargaison</span>
                    <span class="facteur-val">{poids_cargaison:,.0f} kg</span>
                  </div>
                  <div class="facteur-row">
                    <span class="facteur-icon" style="color:{vs_col}">{'▲' if vs_moy>0 else '▼'}</span>
                    <span class="facteur-label">Vs. moyenne port</span>
                    <span class="facteur-val" style="color:{vs_col}">{vs_sign}{vs_moy:.0f} min (moy.=768)</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="glass-card-title" style="margin-bottom:8px;"><i class="fa-solid fa-clock-rotate-left"></i> Historique de la session</div>', unsafe_allow_html=True)
    
    if not st.session_state.historique:
        st.markdown('<p style="color:rgba(255,255,255,0.3);font-size:0.85rem;text-align:center;padding:20px;">Aucune prédiction effectuée</p>', unsafe_allow_html=True)
    else:
        rows = ""
        for h in reversed(st.session_state.historique[-10:]):
            tag_cls = "tag-low" if h["risque"] == "Faible" else "tag-medium" if h["risque"] == "Modéré" else "tag-high"
            rows += f"""<tr>
              <td>{h['timestamp']}</td>
              <td>{h['cargo']}</td><td>{h['heure']}h</td>
              <td>{h['poids']:,.0f} kg</td>
              <td><b style="color:white;">{h['pred']:.0f} min</b></td>
              <td><span class="{tag_cls}">{h['risque']}</span></td>
            </tr>"""
        st.markdown(f"""
        <table class="hist-table">
          <thead><tr><th>Heure</th><th>Cargo</th><th>Heure entrée</th><th>Poids</th><th>Prédiction</th><th>Risque</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

        # Bouton d'export
        col_export1, col_export2, col_export3 = st.columns([1, 1, 2])
        with col_export1:
            if st.button("📥 Export CSV"):
                df_hist = pd.DataFrame(st.session_state.historique)
                csv = df_hist.to_csv(index=False)
                st.download_button(
                    label="Télécharger CSV",
                    data=csv,
                    file_name=f"historique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON PAGE — HISTOGRAMMES PLOTLY
# ═══════════════════════════════════════════════════════════════════════════════

def render_compare_page():
    """Page de comparaison avec histogrammes interactifs."""
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow"><i class="fa-solid fa-chart-bar"></i> Évaluation des algorithmes</div>
        <h1 class="page-header-title">Comparaison des modèles ML</h1>
        <p class="page-header-sub">Performance des 4 algorithmes sur le jeu de test indépendant — 3 187 observations jamais vues.</p>
    </div>
    """, unsafe_allow_html=True)

    # Données pour Plotly
    models = [m[0].replace(" ★", "") for m in MODELES_DATA]
    r2_scores = [m[1] for m in MODELES_DATA]
    mae_scores = [m[2] for m in MODELES_DATA]
    rmse_scores = [m[3] for m in MODELES_DATA]
    colors = [m[4] for m in MODELES_DATA]

    # Histogrammes Plotly
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_r2 = go.Figure(data=[
            go.Bar(x=models, y=r2_scores, marker=dict(color=colors),
                   text=[f"{r2:.3f}" for r2 in r2_scores],
                   textposition='outside',
                   hovertemplate='<b>%{x}</b><br>R² = %{y:.3f}<extra></extra>')
        ])
        fig_r2.update_layout(
            title="📊 Score R² (Plus haut = Mieux)",
            xaxis_title="Modèle",
            yaxis_title="R² Score",
            height=400,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0.1)",
            paper_bgcolor="rgba(10,15,26,0)",
            font=dict(color="rgba(255,255,255,0.7)", size=11),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        fig_mae = go.Figure(data=[
            go.Bar(x=models, y=mae_scores, marker=dict(color=colors),
                   text=[f"{m:.0f}" for m in mae_scores],
                   textposition='outside',
                   hovertemplate='<b>%{x}</b><br>MAE = %{y:.0f} min<extra></extra>')
        ])
        fig_mae.update_layout(
            title="⏱️ MAE (Plus bas = Mieux)",
            xaxis_title="Modèle",
            yaxis_title="MAE (minutes)",
            height=400,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0.1)",
            paper_bgcolor="rgba(10,15,26,0)",
            font=dict(color="rgba(255,255,255,0.7)", size=11),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    with col3:
        fig_rmse = go.Figure(data=[
            go.Bar(x=models, y=rmse_scores, marker=dict(color=colors),
                   text=[f"{r:.0f}" for r in rmse_scores],
                   textposition='outside',
                   hovertemplate='<b>%{x}</b><br>RMSE = %{y:.0f} min<extra></extra>')
        ])
        fig_rmse.update_layout(
            title="📈 RMSE (Plus bas = Mieux)",
            xaxis_title="Modèle",
            yaxis_title="RMSE (minutes)",
            height=400,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0.1)",
            paper_bgcolor="rgba(10,15,26,0)",
            font=dict(color="rgba(255,255,255,0.7)", size=11),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("---")

    # Radar chart
    st.markdown('<div class="glass-card"><div class="glass-card-title"><i class="fa-solid fa-chart-radar"></i> Vue globale — Radar</div>', unsafe_allow_html=True)
    
    fig_radar = go.Figure(data=[
        go.Scatterpolar(
            r=[100, 88, 73, 24],
            theta=['R² (perf)', 'MAE (précision)', 'RMSE (robustesse)', 'Généralisation'],
            fill='toself',
            name='Forêt Aléatoire',
            line=dict(color='#22c55e'),
            fillcolor='rgba(34,197,94,0.2)'
        ),
        go.Scatterpolar(
            r=[88, 82, 87, 80],
            theta=['R² (perf)', 'MAE (précision)', 'RMSE (robustesse)', 'Généralisation'],
            fill='toself',
            name='KNN',
            line=dict(color='#f97316'),
            fillcolor='rgba(249,115,22,0.15)'
        ),
        go.Scatterpolar(
            r=[73, 58, 68, 60],
            theta=['R² (perf)', 'MAE (précision)', 'RMSE (robustesse)', 'Généralisation'],
            fill='toself',
            name='Arbre de Décision',
            line=dict(color='#eab308'),
            fillcolor='rgba(234,179,8,0.15)'
        ),
        go.Scatterpolar(
            r=[24, 0, 0, 50],
            theta=['R² (perf)', 'MAE (précision)', 'RMSE (robustesse)', 'Généralisation'],
            fill='toself',
            name='Régression Linéaire',
            line=dict(color='#f43f5e'),
            fillcolor='rgba(244,63,94,0.15)'
        ),
    ])
    fig_radar.update_layout(
        height=450,
        template="plotly_dark",
        paper_bgcolor="rgba(10,15,26,0)",
        font=dict(color="rgba(255,255,255,0.7)", size=11),
        showlegend=True,
        polar=dict(
            bgcolor="rgba(0,0,0,0.1)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.05)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Analyse
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
      <div class="glass-card-title"><i class="fa-solid fa-lightbulb"></i> Analyse comparative</div>
      <p style="font-size:0.88rem;color:rgba(255,255,255,0.55);line-height:1.8;margin:0;">
        La <b style="color:#4ade80;">Forêt Aléatoire</b> domine sur tous les critères (R²=0.667, MAE=260 min).
        La Régression Linéaire échoue à capturer la complexité non-linéaire du phénomène (R²=0.157).
        Le KNN offre une alternative honorable (R²=0.587) avec un temps d'entraînement plus rapide.
        L'Arbre de Décision souffre d'overfitting (R²=0.490).
      </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def render_analytics_page():
    """Page d'analytics avec graphiques EDA."""
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow"><i class="fa-solid fa-chart-line"></i> Exploration des données</div>
        <h1 class="page-header-title">Analytics du port</h1>
        <p class="page-header-sub">Statistiques clés et insights sur les opérations du Doraleh Port.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Heures d'entrée
        fig_heures = go.Figure(data=[
            go.Bar(
                x=[h[0] for h in DATA_HEURES],
                y=[h[1] for h in DATA_HEURES],
                marker=dict(color='#1d9e75'),
                text=[f"{h[1]:,}" for h in DATA_HEURES],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>%{y:,} camions<extra></extra>'
            )
        ])
        fig_heures.update_layout(
            title="📊 Camions par heure d'entrée",
            xaxis_title="Heure",
            yaxis_title="Nombre de camions",
            height=400,
            template="plotly_dark",
            paper_bgcolor="rgba(10,15,26,0)",
            plot_bgcolor="rgba(0,0,0,0.1)",
            font=dict(color="rgba(255,255,255,0.7)", size=10),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_heures, use_container_width=True)

    with col2:
        # Cargo
        fig_cargo = go.Figure(data=[
            go.Bar(
                y=[c[0] for c in DATA_CARGO],
                x=[c[1] for c in DATA_CARGO],
                orientation='h',
                marker=dict(color=[c[2] for c in DATA_CARGO]),
                text=[f"{c[1]:,}" for c in DATA_CARGO],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>%{x:,} min<extra></extra>'
            )
        ])
        fig_cargo.update_layout(
            title="📦 Surestarie moyenne par cargo",
            xaxis_title="Durée moyenne (minutes)",
            yaxis_title="Type de cargaison",
            height=400,
            template="plotly_dark",
            paper_bgcolor="rgba(10,15,26,0)",
            plot_bgcolor="rgba(0,0,0,0.1)",
            font=dict(color="rgba(255,255,255,0.7)", size=10),
            margin=dict(l=120, r=50, t=50, b=50)
        )
        st.plotly_chart(fig_cargo, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def render_about_page():
    """Page À propos."""
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow"><i class="fa-solid fa-circle-info"></i> À propos du projet</div>
        <h1 class="page-header-title">Port Intelligence AI</h1>
        <p class="page-header-sub">Projet de fin de stage Master 1 IA — Université de Djibouti · Mars–Juin 2026</p>
    </div>

    <div style="display:flex;flex-wrap:wrap;gap:8px;margin:1.5rem 0 2rem;">
      <span style="display:inline-flex;align-items:center;gap:7px;background:rgba(29,158,117,0.1);border:1px solid rgba(29,158,117,0.25);color:#4ade80;padding:5px 14px;border-radius:99px;font-size:0.78rem;font-weight:600;"><i class="fa-solid fa-database"></i> 62 329 enregistrements</span>
      <span style="display:inline-flex;align-items:center;gap:7px;background:rgba(29,158,117,0.1);border:1px solid rgba(29,158,117,0.25);color:#4ade80;padding:5px 14px;border-radius:99px;font-size:0.78rem;font-weight:600;"><i class="fa-solid fa-tree"></i> R²=0.667</span>
      <span style="display:inline-flex;align-items:center;gap:7px;background:rgba(29,158,117,0.1);border:1px solid rgba(29,158,117,0.25);color:#4ade80;padding:5px 14px;border-radius:99px;font-size:0.78rem;font-weight:600;"><i class="fa-solid fa-python"></i> Python/Scikit-learn</span>
    </div>

    <div class="glass-card">
      <div class="glass-card-title"><i class="fa-solid fa-book"></i> Méthodologie</div>
      <p style="font-size:0.9rem;color:rgba(255,255,255,0.6);line-height:1.8;margin:0;">
        <strong>1. Collecte:</strong> 62 329 enregistrements réels (oct. 2023 – juin 2024)<br>
        <strong>2. Nettoyage:</strong> Suppression NaN, outliers, data leakage<br>
        <strong>3. Feature Engineering:</strong> Variables temporelles, transformations log<br>
        <strong>4. Modélisation:</strong> 4 algorithmes testés, Forêt Aléatoire ★<br>
        <strong>5. Déploiement:</strong> Streamlit, Gradio, Hugging Face Spaces
      </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

def render_footer():
    """Afficher le footer."""
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        uri = img_to_b64(logo_path)
        logo_html = f'<img src="{uri}" style="width:100%;height:100%;object-fit:contain;border-radius:8px;" />'
    else:
        logo_html = '<i class="fa-solid fa-anchor"></i>'

    st.markdown(f"""
    <div class="site-footer">
      <div class="footer-grid">
        <div>
          <div class="footer-brand-row">
            <div class="footer-logo-box">{logo_html}</div>
            <span class="footer-app-name">Port<span>Intelligence</span> AI</span>
          </div>
          <p class="footer-desc">Prédiction de la surestarie par ML sur données réelles du Doraleh Port.</p>
        </div>
        <div>
          <div class="footer-col-title">Navigation</div>
          <a class="footer-link" href="?page=home" target="_self"><i class="fa-solid fa-anchor"></i> Accueil</a>
          <a class="footer-link" href="?page=predict" target="_self"><i class="fa-solid fa-clock"></i> Prédiction</a>
          <a class="footer-link" href="?page=compare" target="_self"><i class="fa-solid fa-chart-bar"></i> Comparaison</a>
          <a class="footer-link" href="?page=analytics" target="_self"><i class="fa-solid fa-chart-line"></i> Analytics</a>
          <a class="footer-link" href="?page=about" target="_self"><i class="fa-solid fa-circle-info"></i> À propos</a>
        </div>
        <div>
          <div class="footer-col-title">Projet</div>
          <a class="footer-link" href="https://github.com/HASSAN-MAHAMOUD/port-intelligence-ai" target="_blank"><i class="fa-brands fa-github"></i> Code source</a>
          <a class="footer-link" href="https://huggingface.co/spaces/HassanML2026/port-intelligence-ai" target="_blank"><i class="fa-solid fa-rocket"></i> Hugging Face</a>
        </div>
      </div>
      <div class="footer-bottom">
        <div class="footer-copy">&copy; 2026 <span>Hassan Mahamoud</span> — Master 1 IA · Université de Djibouti</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Fonction principale."""
    apply_css_styles()
    render_navbar()

    # Router
    if st.session_state.page == "home":
        render_home_page()
    elif st.session_state.page == "predict":
        render_predict_page()
    elif st.session_state.page == "compare":
        render_compare_page()
    elif st.session_state.page == "analytics":
        render_analytics_page()
    elif st.session_state.page == "about":
        render_about_page()

    st.markdown('</div>', unsafe_allow_html=True)
    render_footer()

if __name__ == "__main__":
    main()
