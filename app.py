import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import os
import base64
import plotly.graph_objects as go

# ── Configuration page ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Port Intelligence AI — DMP",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Helper : encode image file to base64 ──────────────────────────────────────
def img_to_b64(path):
    if not os.path.exists(path):
        return None
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp"}.get(ext, "image/png")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

# ── Chargement du modèle ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "meilleur_modele_foret_simple.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

pipe = load_model()

if pipe is not None:
    cat_step          = pipe.named_steps['comp'].named_transformers_['cat']
    ohe               = cat_step.named_steps['onehot']
    imp_cat           = cat_step.named_steps['imputer']
    NOM_CARGAISON     = sorted(ohe.categories_[0].tolist())
    NOM_NAVIRE        = sorted(ohe.categories_[1].tolist())
    TYPE_TRAVAIL      = sorted(ohe.categories_[4].tolist())
    OP_ENTREE_DEFAULT = imp_cat.statistics_[2]
    OP_SORTIE_DEFAULT = imp_cat.statistics_[3]
else:
    NOM_CARGAISON     = ['Fertilizer','Steel Bar','Steel Coil','Vehicle','Wheat','Cement','Sugar','Rice','Clinker','General Cargo']
    NOM_NAVIRE        = ['ABBAY II','GALINI','AFEA']
    TYPE_TRAVAIL      = ['Delivery','Direct Discharge','Receiving']
  

# ── Données EDA statiques ─────────────────────────────────────────────────────
MODELES_DATA = [
    ("Régression Linéaire", 0.157, 480, 780,  "#f43f5e"),
    ("KNN",                 0.587, 246, 546,  "#f97316"),
    ("Arbre de Décision",   0.490, 353, 607,  "#eab308"),
    ("Forêt Aléatoire ★",  0.667, 260, 552,  "#22c55e"),
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

# ── CSS Global ────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
:root {
  --accent:      #1d9e75;
  --accent-2:    #0a2540;
  --accent-glow: rgba(29,158,117,0.2);
  --bg:          #06090f;
  --bg-2:        #0a0f1a;
  --surface:     rgba(255,255,255,0.04);
  --surface-h:   rgba(255,255,255,0.07);
  --border:      rgba(255,255,255,0.08);
  --border-h:    rgba(29,158,117,0.4);
  --text:        #ffffff;
  --text-muted:  rgba(255,255,255,0.5);
  --text-dim:    rgba(255,255,255,0.25);
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
.navbar a, .navbar a:hover, .navbar a:visited,
.navbar a:focus, .navbar a:active { text-decoration: none !important; }
.navbar {
    position: fixed; top:0; left:0; right:0; z-index:9999;
    display:flex; align-items:center; justify-content:space-between;
    padding:0 2.5rem; height:64px;
    background:rgba(6,9,15,0.88);
    backdrop-filter:blur(20px);
    border-bottom:1px solid rgba(255,255,255,0.07);
}
.navbar-logo { display:flex; align-items:center; gap:10px; cursor:pointer; }
.navbar-logo-icon {
    width:40px; height:40px;
    background:#ffffff;
    border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:1rem; color:white; overflow:hidden;
    box-shadow:0 2px 10px rgba(0,0,0,0.25);
}
.navbar-logo-icon img { width:100%; height:100%; object-fit:contain; }
.navbar-logo-text {
    font-family:'Syne',sans-serif; font-size:1.1rem;
    font-weight:800; color:#fff; letter-spacing:0.02em;
}
.navbar-logo-text span { color:#1d9e75; }
.navbar-links { display:flex; align-items:center; gap:0.3rem; }
.nav-btn {
    display:inline-flex; align-items:center; gap:8px;
    padding:7px 16px; border-radius:8px;
    font-size:0.85rem; font-weight:500;
    color:rgba(255,255,255,0.55);
    border:none; background:transparent;
    transition:all 0.2s; white-space:nowrap; text-decoration:none;
}
.nav-btn:hover, .nav-btn.active { color:#fff; background:rgba(255,255,255,0.07); }
.nav-btn.active { color:#4ade80; }
.nav-btn i { font-size:0.78rem; }

/* ══ PAGE WRAPPER ══ */
.page-wrapper {
    padding-top:80px; max-width:1200px;
    margin:0 auto; padding-left:2rem; padding-right:2rem;
}

/* ══ HOME PAGE ══ */
.hero-section { padding:5rem 0 1.5rem; text-align:center; }
.hero-pill {
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(29,158,117,0.12); border:1px solid rgba(29,158,117,0.3);
    border-radius:99px; padding:5px 16px;
    font-size:0.78rem; font-weight:600; color:#4ade80;
    letter-spacing:0.05em; text-transform:uppercase; margin-bottom:1.5rem;
}
.hero-title {
    font-family:'Syne',sans-serif;
    font-size:clamp(3.2rem,8vw,6rem); font-weight:800;
    color:#fff; line-height:1.05; margin:0 0 0.8rem;
}
.hero-title .gradient-text {
    background:linear-gradient(135deg,#1d9e75 0%,#22d3ee 50%,#3b82f6 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-subtitle-inline {
    font-size:1rem; color:rgba(255,255,255,0.45); max-width:620px;
    margin:0 auto 1.8rem; line-height:1.7; font-weight:300;
}
.stats-band {
    display:grid; grid-template-columns:repeat(4,1fr); gap:1.2rem;
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius:24px; padding:2.8rem 2rem; margin-bottom:4rem; text-align:center;
}
.stat-num {
    font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
    background:linear-gradient(135deg,#1d9e75,#22d3ee);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.stat-lbl { font-size:0.88rem; color:rgba(255,255,255,0.4); margin-top:0.4rem; }

/* ══ PAGE HEADER ══ */
.page-header { padding:3rem 0 2rem; }
.page-header-eyebrow {
    font-size:0.75rem; font-weight:600; text-transform:uppercase;
    letter-spacing:0.12em; color:#1d9e75; margin-bottom:0.6rem;
    display:flex; align-items:center; gap:8px;
}
.page-header-title {
    font-family:'Syne',sans-serif; font-size:2.2rem;
    font-weight:800; color:white; margin:0 0 0.6rem;
}
.page-header-sub { font-size:0.95rem; color:rgba(255,255,255,0.45); }

/* ══ GLASS CARD ══ */
.glass-card {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:20px; padding:1.5rem; margin-bottom:1rem;
}
.glass-card-title {
    font-size:0.72rem; font-weight:600; text-transform:uppercase;
    letter-spacing:0.1em; color:rgba(255,255,255,0.3); margin-bottom:1rem;
    display:flex; align-items:center; gap:8px;
}

/* ══ RESULT HERO ══ */
.result-hero {
    background:linear-gradient(135deg,rgba(29,158,117,0.12),rgba(59,130,246,0.12));
    border:1px solid rgba(29,158,117,0.25); border-radius:20px;
    padding:2.5rem 1.5rem; text-align:center; margin-bottom:1rem;
    position:relative; overflow:hidden;
}
.result-hero::after {
    content:''; position:absolute; top:-60px; right:-60px;
    width:180px; height:180px; border-radius:50%;
    background:radial-gradient(circle,rgba(29,158,117,0.12),transparent 70%);
    pointer-events:none;
}
.result-time {
    font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800;
    color:white; margin:0;
}
.result-hmin {
    font-size:1.1rem; font-weight:500;
    background:linear-gradient(135deg,#1d9e75,#22d3ee);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin:0.3rem 0 0;
}
.result-sub { font-size:0.85rem; color:rgba(255,255,255,0.4); }

/* ══ GAUGE ══ */
.gauge-wrap { padding:1.2rem 0 0.6rem; }
.gauge-bar-bg {
    height:14px; border-radius:99px;
    background:linear-gradient(90deg,#16a34a 0%,#16a34a 40%,#d97706 40%,#d97706 70%,#dc2626 70%,#dc2626 100%);
    position:relative; margin-bottom:6px;
}
.gauge-needle {
    position:absolute; top:-5px;
    width:4px; height:24px;
    background:white; border-radius:2px;
    transform:translateX(-50%);
    box-shadow:0 0 8px rgba(255,255,255,0.4);
    transition:left 0.8s cubic-bezier(.34,1.56,.64,1);
}
.gauge-labels {
    display:flex; justify-content:space-between;
    font-size:0.68rem; color:rgba(255,255,255,0.3);
}
.risk-badge-wrap { text-align:center; margin-top:12px; }
.risk-badge {
    display:inline-block; font-size:0.9rem; font-weight:700;
    padding:7px 22px; border-radius:99px;
}
.risk-low    { background:rgba(22,163,74,0.2);  border:1px solid rgba(22,163,74,0.4);  color:#4ade80; }
.risk-medium { background:rgba(217,119,6,0.2);  border:1px solid rgba(217,119,6,0.4);  color:#fbbf24; }
.risk-high   { background:rgba(220,38,38,0.2);  border:1px solid rgba(220,38,38,0.4);  color:#f87171; }

/* ══ FACTEURS ══ */
.facteur-row {
    display:flex; align-items:center; gap:10px; padding:9px 0;
    border-bottom:1px solid rgba(255,255,255,0.06);
}
.facteur-row:last-child { border-bottom:none; }
.facteur-icon { font-size:0.8rem; width:18px; text-align:center; }
.facteur-label { flex:1; font-size:0.83rem; font-weight:500; color:rgba(255,255,255,0.7); }
.facteur-val { font-size:0.83rem; color:rgba(255,255,255,0.4); }

/* ══ HISTORIQUE ══ */
.hist-table { width:100%; border-collapse:collapse; font-size:0.82rem; margin-top:8px; }
.hist-table th {
    background:rgba(255,255,255,0.05); color:rgba(255,255,255,0.3);
    font-weight:600; padding:8px 10px; text-align:left;
    border-bottom:1px solid rgba(255,255,255,0.07);
    font-size:0.68rem; text-transform:uppercase; letter-spacing:0.05em;
}
.hist-table td { padding:7px 10px; border-bottom:1px solid rgba(255,255,255,0.04); color:rgba(255,255,255,0.7); }
.hist-table tr:last-child td { border-bottom:none; }
.tag-low    { background:rgba(22,163,74,0.15);  color:#4ade80;  padding:2px 9px; border-radius:99px; font-size:0.72rem; font-weight:600; }
.tag-medium { background:rgba(217,119,6,0.15);  color:#fbbf24;  padding:2px 9px; border-radius:99px; font-size:0.72rem; font-weight:600; }
.tag-high   { background:rgba(220,38,38,0.15);  color:#f87171;  padding:2px 9px; border-radius:99px; font-size:0.72rem; font-weight:600; }

/* ══ COMPARAISON MODELES ══ */
.model-card {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:20px; padding:1.5rem; margin-bottom:12px; position:relative;
    transition:border-color var(--tr), transform var(--tr);
}
.model-card:hover { transform:translateY(-3px); }
.model-card.best {
    border:1px solid rgba(29,158,117,0.4);
    background:linear-gradient(135deg,rgba(29,158,117,0.08),rgba(59,130,246,0.05));
}
.model-name {
    font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
    color:white; margin:0 0 1rem;
}
.best-pill {
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(29,158,117,0.2); border:1px solid rgba(29,158,117,0.4);
    color:#4ade80; padding:3px 12px; border-radius:99px;
    font-size:0.7rem; font-weight:700; margin-left:10px;
    vertical-align:middle;
}
.metric-row { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.metric-lbl { font-size:0.75rem; color:rgba(255,255,255,0.35); width:46px; flex-shrink:0; }
.metric-track { flex:1; height:6px; background:rgba(255,255,255,0.06); border-radius:99px; overflow:hidden; }
.metric-fill { height:100%; border-radius:99px; }
.metric-val { font-size:0.78rem; font-weight:600; color:rgba(255,255,255,0.6); width:60px; text-align:right; flex-shrink:0; }

/* ══ BAR CHART EDA ══ */
.bar-row { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.bar-label { width:68px; font-size:0.78rem; color:rgba(255,255,255,0.4); text-align:right; flex-shrink:0; }
.bar-track { flex:1; height:8px; background:rgba(255,255,255,0.06); border-radius:99px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:99px; }
.bar-num   { width:58px; font-size:0.78rem; color:rgba(255,255,255,0.35); text-align:right; flex-shrink:0; }

/* ══ ABOUT / PIPELINE ══ */
.about-eyebrow {
    display:inline-flex; align-items:center; gap:8px;
    font-size:0.7rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.14em; color:#1d9e75; margin-bottom:0.9rem;
}
.about-section-title {
    font-family:'Syne',sans-serif;
    font-size:clamp(1.6rem,3.5vw,2.4rem); font-weight:800;
    color:white; line-height:1.15; margin:0 0 0.8rem;
}
.about-section-title .grad {
    background:linear-gradient(120deg,#1d9e75 0%,#22d3ee 55%,#3b82f6 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.about-sub {
    font-size:0.95rem; color:rgba(255,255,255,0.5);
    font-weight:300; max-width:560px; line-height:1.8;
}
.a-divider { border:none; border-top:1px solid rgba(255,255,255,0.07); margin:3rem 0; }

.pipeline-step {
    display:flex; align-items:flex-start; gap:16px;
    padding:16px 0; border-bottom:1px solid rgba(255,255,255,0.06);
}
.pipeline-step:last-child { border-bottom:none; }
.step-num {
    width:34px; height:34px; border-radius:50%;
    background:linear-gradient(135deg,#1d9e75,#0a7a5a);
    display:flex; align-items:center; justify-content:center;
    font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:800;
    color:white; flex-shrink:0;
}
.step-title { font-size:0.92rem; font-weight:600; color:white; margin:0 0 4px; }
.step-desc  { font-size:0.82rem; color:rgba(255,255,255,0.45); margin:0; line-height:1.6; }

.arch-duo { display:grid; grid-template-columns:1fr 1fr; gap:1.4rem; margin-top:2rem; }
.arch-card {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:var(--r-lg); overflow:hidden;
    transition:transform var(--tr), box-shadow var(--tr), border-color var(--tr);
}
.arch-card:hover { transform:translateY(-6px); }
.arch-card.rf:hover  { border-color:rgba(29,158,117,0.45); box-shadow:0 24px 60px rgba(29,158,117,0.18); }
.arch-card.knn:hover { border-color:rgba(249,115,22,0.45); box-shadow:0 24px 60px rgba(249,115,22,0.18); }
.arch-placeholder {
    height:140px; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:0.5rem; padding:1rem;
}
.arch-placeholder i { font-size:2.2rem; }
.arch-placeholder p { font-size:0.72rem; color:rgba(255,255,255,0.25); margin:0; text-align:center; }
.arch-body { padding:1.4rem; }
.arch-badge {
    display:inline-flex; align-items:center; gap:6px;
    padding:4px 12px; border-radius:99px;
    font-size:0.68rem; font-weight:700; letter-spacing:0.05em;
    text-transform:uppercase; margin-bottom:0.8rem;
}
.badge-rf  { background:rgba(29,158,117,0.15); border:1px solid rgba(29,158,117,0.3); color:#4ade80; }
.badge-knn { background:rgba(249,115,22,0.15); border:1px solid rgba(249,115,22,0.3); color:#fb923c; }
.arch-title { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:800; color:white; margin-bottom:0.7rem; }
.arch-desc  { font-size:0.84rem; color:rgba(255,255,255,0.5); line-height:1.75; font-weight:300; margin-bottom:1rem; }
.arch-spec  { display:flex; align-items:flex-start; gap:8px; font-size:0.8rem; color:rgba(255,255,255,0.5); margin-bottom:6px; }
.arch-spec i { font-size:0.65rem; margin-top:4px; flex-shrink:0; }

.tech-6 { display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin-top:2rem; }
.tech-card {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.07);
    border-radius:var(--r-md); padding:1.4rem 0.7rem 1.1rem;
    text-align:center; cursor:default; position:relative; overflow:hidden;
    transition:border-color var(--tr), background var(--tr), transform var(--tr), box-shadow var(--tr);
}
.tech-card:hover { transform:translateY(-6px); border-color:rgba(29,158,117,0.35); background:rgba(255,255,255,0.07); }
.tech-icon { width:48px; height:48px; margin:0 auto 0.75rem; border-radius:var(--r-sm); display:flex; align-items:center; justify-content:center; font-size:1.5rem; transition:transform var(--tr); }
.tech-card:hover .tech-icon { transform:scale(1.12) rotate(-4deg); }
.tech-name { font-family:'Syne',sans-serif; font-size:0.8rem; font-weight:700; color:white; }
.tech-role { font-size:0.65rem; color:rgba(255,255,255,0.28); margin-top:2px; }
.tech-bar { position:absolute; bottom:0; left:0; right:0; height:2px; border-radius:99px; transform:scaleX(0); transition:transform var(--tr); }
.tech-card:hover .tech-bar { transform:scaleX(1); }

/* ══ FOOTER ══ */
.site-footer { border-top:1px solid rgba(255,255,255,0.07); padding:2.8rem 0 1.8rem; margin-top:4rem; }
.site-footer a, .site-footer a:hover, .site-footer a:visited,
.site-footer a:focus, .site-footer a:active { text-decoration:none !important; }
.footer-grid { display:grid; grid-template-columns:1.4fr 1fr 1fr; gap:3rem; margin-bottom:2.2rem; }
.footer-brand-row { display:flex; align-items:center; gap:10px; margin-bottom:0.9rem; }
.footer-logo-box {
    width:40px; height:40px;
    background:#ffffff;
    border-radius:10px; display:flex; align-items:center; justify-content:center;
    font-size:14px; color:white; overflow:hidden; flex-shrink:0;
    box-shadow:0 2px 10px rgba(0,0,0,0.2);
}
.footer-app-name { font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; color:white; }
.footer-app-name span { color:#1d9e75; }
.footer-desc { font-size:0.82rem; color:rgba(255,255,255,0.38); line-height:1.7; font-weight:300; max-width:240px; }
.footer-col-title { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:rgba(255,255,255,0.22); margin-bottom:0.9rem; }
.footer-link {
    display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:8px;
    font-size:0.82rem; font-weight:500; color:rgba(255,255,255,0.42);
    text-decoration:none !important; margin-bottom:0.2rem;
    transition:color 0.2s, background 0.2s; width:100%;
}
.footer-link i { font-size:0.72rem; flex-shrink:0; }
.footer-link:hover { color:#fff; background:rgba(255,255,255,0.07); }
.footer-bottom { padding-top:1.4rem; border-top:1px solid rgba(255,255,255,0.07); display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.8rem; }
.footer-copy { font-size:0.74rem; color:rgba(255,255,255,0.22); }
.footer-copy span { color:#1d9e75; }

/* ══ Streamlit overrides ══ */
.stApp > div { background: transparent !important; }
.stSelectbox > div > div {
    background:rgba(255,255,255,0.05) !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:10px !important; color:white !important;
}
.stSelectbox label, .stSlider label, .stNumberInput label {
    color:rgba(255,255,255,0.65) !important; font-size:0.85rem !important;
}
.stSlider > div > div > div { background:linear-gradient(135deg,#1d9e75,#22d3ee) !important; }
.stNumberInput > div > div > input {
    background:rgba(255,255,255,0.05) !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:10px !important; color:white !important;
}
div[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#1d9e75,#0f6e56) !important;
    color:white !important; border:none !important;
    border-radius:12px !important; font-size:0.95rem !important;
    font-weight:600 !important; padding:12px 28px !important;
    width:100% !important;
}
div[data-testid="stButton"] > button:hover { opacity:0.9 !important; }
hr { border-color:rgba(255,255,255,0.07) !important; }
div[data-testid="column"] { padding:0 0.4rem !important; }

@media(max-width:900px){
    .arch-duo { grid-template-columns:1fr; }
    .tech-6 { grid-template-columns:repeat(3,1fr); }
    .footer-grid { grid-template-columns:1fr; gap:1.8rem; }
    .stats-band { grid-template-columns:repeat(2,1fr); }
}
             
/* ══ HOW IT WORKS ══ */
.hiw-section{margin:3rem 0;}
.hiw-title{
    font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
    color:white;text-align:center;margin-bottom:0.5rem;
}
.hiw-sub{
    font-size:0.9rem;color:rgba(255,255,255,0.4);
    text-align:center;margin-bottom:2rem;
}
.hiw-steps{
    display:grid;grid-template-columns:repeat(3,1fr);gap:1.5rem;
    position:relative;
}
.hiw-step{
    background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
    border-radius:20px;padding:2rem 1.5rem;text-align:center;
    position:relative;transition:border-color var(--tr),transform var(--tr);
}
.hiw-step:hover{border-color:rgba(29,158,117,0.35);transform:translateY(-4px);}
.hiw-num{
    width:40px;height:40px;border-radius:50%;
    background:linear-gradient(135deg,#1d9e75,#22d3ee);
    display:flex;align-items:center;justify-content:center;
    font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;color:white;
    margin:0 auto 1.2rem;
}
.hiw-icon{font-size:2rem;color:#1d9e75;margin-bottom:0.8rem;}
.hiw-step-title{font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:white;margin-bottom:0.5rem;}
.hiw-step-desc{font-size:0.82rem;color:rgba(255,255,255,0.45);line-height:1.65;}
.hiw-arrow{
    position:absolute;top:50%;right:-18px;
    font-size:1.2rem;color:rgba(29,158,117,0.5);
    transform:translateY(-50%);z-index:2;
}
 
</style>
""", unsafe_allow_html=True)


# ── Session state & routing ───────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "historique" not in st.session_state:
    st.session_state.historique = []

VALID_PAGES = ["home", "predict", "compare", "about"]
if "page" in st.query_params:
    p = st.query_params["page"]
    if p in VALID_PAGES:
        st.session_state.page = p


# ── Navbar ────────────────────────────────────────────────────────────────────
pages_nav = [
    ("home",    "fa-anchor",      "Accueil"),
    ("predict", "fa-clock",       "Prédiction"),
    ("compare", "fa-chart-bar",   "Comparaison"),
    ("about",   "fa-circle-info", "À propos"),
]

def get_logo_html():
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        uri = img_to_b64(logo_path)
        return f'<img src="{uri}" style="width:100%;height:100%;object-fit:contain;padding:4px;" />'
    return '<i class="fa-solid fa-anchor"></i>'

nav_links = ""
for pid, icon, label in pages_nav:
    active_cls = "active" if st.session_state.page == pid else ""
    nav_links += f'<a class="nav-btn {active_cls}" href="?page={pid}" target="_self"><i class="fa-solid {icon}"></i> {label}</a>'

st.markdown(f"""
<div class="navbar">
  <div class="navbar-logo">
    <div class="navbar-logo-icon">{get_logo_html()}</div>
    <span class="navbar-logo-text">Port<span>Intelligence</span></span>
  </div>
  <div class="navbar-links">{nav_links}</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    st.markdown("""
    <div class="hero-section">
        <div class="hero-pill"><i class="fa-solid fa-ship"></i> Doraleh Multi-Purpose Port · Djibouti</div>
        <h1 class="hero-title">
            Prédire la surestarie<br>au <span class="gradient-text">Doraleh Multi-Purpose Port</span>
        </h1>
   
    </div>
    """, unsafe_allow_html=True)

    # Carousel
    slides_data = [
        (1, "fa-clock",      "Prédiction de la surestarie",
         "Estimez instantanément la durée d'attente d'un camion au terminal DMP à partir de ses caractéristiques à la Gate In.",
         "PRÉDIRE · ANTICIPER · OPTIMISER",
         "linear-gradient(135deg,rgba(29,158,117,0.25),rgba(59,130,246,0.2))"),
        (2, "fa-tree",       "Forêt Aléatoire — Modèle retenu",
         "100 arbres de décision entraînés sur 25 498 observations. R²=0.667 sur le jeu de test, MAE=260 minutes.",
         "R²=0.667 · MAE=260 MIN · RMSE=552 MIN",
         "linear-gradient(135deg,rgba(34,211,238,0.2),rgba(29,158,117,0.2))"),
        (3, "fa-chart-bar",  "Comparaison de 4 algorithmes",
         "Régression Linéaire, KNN, Arbre de Décision et Forêt Aléatoire évalués et comparés sur les mêmes données DMP.",
         "4 MODÈLES · SPLIT 80/10/10 · JUSTE COMPARAISON",
         "linear-gradient(135deg,rgba(249,115,22,0.2),rgba(59,130,246,0.2))"),
        (4, "fa-database",   "62 329 enregistrements réels",
         "Gate_GC, Gate_Out, DISCH et LOAD — données opérationnelles confidentielles fournies par le port DMP/PDSA.",
         "DONNÉES RÉELLES · OCT 2023 – JUIN 2024 · DJIBOUTI",
         "linear-gradient(135deg,rgba(59,130,246,0.2),rgba(29,158,117,0.2))"),
    ]

    slides_html = ""
    for idx, (_, icon, title, desc, slogan, bg) in enumerate(slides_data):
        img_path = f"assets/carousel_{idx+1}.jpg"
        if not os.path.exists(img_path):
            img_path = f"assets/carousel_{idx+1}.png"
        if os.path.exists(img_path):
            uri = img_to_b64(img_path)
            slides_html += f"""
<div class="carousel-slide" style="min-width:100%;position:relative;overflow:hidden;height:400px;">
  <img src="{uri}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;z-index:0;" />
  <div style="position:absolute;inset:0;background:{bg};z-index:1;"></div>
  <div style="position:relative;z-index:2;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem;text-align:center;">
    <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:white;margin-bottom:0.7rem;text-shadow:0 2px 16px rgba(0,0,0,0.5);">{title}</div>
    <div style="font-size:0.95rem;color:rgba(255,255,255,0.75);max-width:520px;margin:0 auto;line-height:1.75;text-shadow:0 1px 8px rgba(0,0,0,0.4);">{desc}</div>
    <div style="margin-top:1.4rem;font-size:0.72rem;font-weight:600;letter-spacing:0.15em;text-transform:uppercase;color:rgba(255,255,255,0.4);">{slogan}</div>
  </div>
</div>"""
        else:
            slides_html += f"""
<div class="carousel-slide" style="background:{bg};min-width:100%;padding:4.5rem 3rem;text-align:center;position:relative;overflow:hidden;">
  <div style="font-size:4rem;color:rgba(29,158,117,0.8);margin-bottom:1.5rem;"><i class="fa-solid {icon}"></i></div>
  <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:white;margin-bottom:0.7rem;">{title}</div>
  <div style="font-size:0.95rem;color:rgba(255,255,255,0.55);max-width:520px;margin:0 auto;line-height:1.75;">{desc}</div>
  <div style="margin-top:1.4rem;font-size:0.72rem;font-weight:600;letter-spacing:0.15em;text-transform:uppercase;color:rgba(255,255,255,0.25);">{slogan}</div>
</div>"""

    carousel_html = f"""<!DOCTYPE html><html><head>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;}}
.carousel-wrap{{position:relative;width:100%;border-radius:28px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.03);}}
.carousel-inner{{display:flex;transition:transform 0.6s cubic-bezier(0.25,1,0.5,1);}}
.carousel-arrow{{position:absolute;top:50%;transform:translateY(-50%);width:44px;height:44px;background:rgba(0,0,0,0.5);border:1px solid rgba(255,255,255,0.1);border-radius:50%;color:white;display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:10;font-size:1rem;transition:background 0.2s;}}
.carousel-arrow:hover{{background:rgba(29,158,117,0.5);}}
.arr-left{{left:18px;}}.arr-right{{right:18px;}}
.dots{{display:flex;justify-content:center;gap:8px;padding:1.2rem 0;background:rgba(0,0,0,0.3);}}
.dot{{width:30px;height:7px;border-radius:99px;background:rgba(255,255,255,0.15);cursor:pointer;border:none;transition:all 0.3s;}}
.dot.active{{background:#1d9e75;width:40px;}}
</style></head><body>
<div class="carousel-wrap">
<div class="carousel-inner" id="ci">{slides_html}</div>
<button class="carousel-arrow arr-left" onclick="prev()"><i class="fa-solid fa-chevron-left"></i></button>
<button class="carousel-arrow arr-right" onclick="next()"><i class="fa-solid fa-chevron-right"></i></button>
<div class="dots" id="dots"></div>
</div>
<script>
let cur=0;const total=4;const dotsEl=document.getElementById('dots');
for(let i=0;i<total;i++){{const b=document.createElement('button');b.className='dot'+(i===0?' active':'');b.id='d'+i;b.onclick=()=>go(i);dotsEl.appendChild(b);}}
function update(){{document.getElementById('ci').style.transform=`translateX(-${{cur*100}}%)`;for(let i=0;i<total;i++)document.getElementById('d'+i).className='dot'+(i===cur?' active':'');}}
function next(){{cur=(cur+1)%total;update();}}function prev(){{cur=(cur-1+total)%total;update();}}function go(i){{cur=i;update();}}setInterval(next,4500);
</script></body></html>"""

    components.html(carousel_html, height=460)
   # ── Comment ça marche ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="hiw-section">
        <div class="about-eyebrow" style="justify-content:center;display:flex;"><i class="fa-solid fa-diagram-project"></i> Fonctionnement</div>
        <div class="hiw-title">Comment ça marche ?</div>
        <div class="hiw-sub">De la Gate In au résultat en 3 étapes</div>
        <div class="hiw-steps">
            <div class="hiw-step">
                <div class="hiw-num">1</div>
                <div class="hiw-icon"><i class="fa-solid fa-clipboard-list"></i></div>
                <div class="hiw-step-title">Renseigner les données</div>
                <div class="hiw-step-desc">Saisissez le poids, l'heure d'entrée, le type de cargaison et le navire à la Gate In du terminal DMP.</div>
                <span class="hiw-arrow"><i class="fa-solid fa-chevron-right"></i></span>
            </div>
            <div class="hiw-step">
                <div class="hiw-num">2</div>
                <div class="hiw-icon"><i class="fa-solid fa-tree"></i></div>
                <div class="hiw-step-title">L'IA analyse</div>
                <div class="hiw-step-desc">La Forêt Aléatoire (100 arbres, entraînée sur 25 498 opérations réelles) prédit la surestarie en millisecondes.</div>
                <span class="hiw-arrow"><i class="fa-solid fa-chevron-right"></i></span>
            </div>
            <div class="hiw-step">
                <div class="hiw-num">3</div>
                <div class="hiw-icon"><i class="fa-solid fa-gauge-high"></i></div>
                <div class="hiw-step-title">Résultat instantané</div>
                <div class="hiw-step-desc">Obtenez la durée estimée, l'intervalle de confiance, le niveau de risque et des conseils opérationnels.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)    

    st.markdown("""

    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "predict":

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

    col_form, col_map = st.columns([3, 2])

    with col_form:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-card"><div class="glass-card-title"><i class="fa-solid fa-weight-hanging"></i> Poids (kg)</div>', unsafe_allow_html=True)
            poids_tare      = st.number_input("Poids Tare (kg)", value=8500, min_value=0, step=500)
            poids_cargaison = st.number_input("Poids Cargaison (kg)", value=15000, min_value=0, step=500)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card"><div class="glass-card-title"><i class="fa-solid fa-calendar"></i> Temporel</div>', unsafe_allow_html=True)
            heure = st.slider("Heure d'entrée (0–23)", 0, 23, 14)
            jour  = st.slider("Jour (0=Lundi · 6=Dimanche)", 0, 6, 1)
            mois  = st.slider("Mois", 1, 12, 3)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><div class="glass-card-title"><i class="fa-solid fa-ship"></i> Opération</div>', unsafe_allow_html=True)
        col_op1, col_op2, col_op3 = st.columns(3)
        with col_op1:
            nom_carg  = st.selectbox("Type de cargaison", NOM_CARGAISON)
        with col_op2:
            nom_nav   = st.selectbox("Navire", NOM_NAVIRE)
        with col_op3:
            type_trav = st.selectbox("Type de travail", TYPE_TRAVAIL)
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("⚓  Prédire la surestarie", use_container_width=True)

        # Inject logo into the button via JS if logo exists
        _logo_b64 = img_to_b64("assets/logo.png")
        if _logo_b64:
            st.markdown(f"""
            <script>
            (function(){{
              var btns = window.parent.document.querySelectorAll('button[kind="primary"], button[data-testid="baseButton-primary"]');
              btns.forEach(function(b){{
                if(b.innerText && b.innerText.includes('Pr\\u00e9dire')) {{
                  b.innerHTML = '<img src="{_logo_b64}" style="width:18px;height:18px;object-fit:contain;vertical-align:middle;margin-right:8px;border-radius:4px;"/> Prédire la surestarie';
                }}
              }});
            }})();
            </script>
            """, unsafe_allow_html=True)

    with col_map:
        _map_logo_b64 = img_to_b64("assets/logo.png")
        _popup_logo = (
            f'<img src="{_map_logo_b64}" style="width:22px;height:22px;object-fit:contain;'
            f'vertical-align:middle;margin-right:6px;border-radius:5px;background:#fff;padding:1px;"/>'
            if _map_logo_b64 else "&#x2693;"
        )
        djibouti_map_html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
html,body{{height:100%;background:transparent;}}
#map{{width:100%;height:590px;border-radius:20px;overflow:hidden;border:1px solid rgba(29,158,117,0.35);}}
.leaflet-popup-content-wrapper{{background:#0a0f1a;border:1px solid rgba(29,158,117,0.4);border-radius:12px;color:white;box-shadow:0 8px 32px rgba(0,0,0,0.6);}}
.leaflet-popup-content{{margin:10px 14px;font-family:'DM Sans',sans-serif;}}
.leaflet-popup-tip{{background:#0a0f1a;}}
.popup-title{{font-weight:700;font-size:0.85rem;color:#4ade80;margin-bottom:4px;display:flex;align-items:center;}}
.popup-sub{{font-size:0.72rem;color:rgba(255,255,255,0.55);line-height:1.5;}}
</style></head><body>
<div id="map"></div>
<script>
var map = L.map('map', {{zoomControl:true, attributionControl:false}}).setView([11.5720, 43.1200], 13);

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom:19, subdomains:'abcd'
}}).addTo(map);

var pulseIcon = L.divIcon({{
  html: '<div style="position:relative;width:48px;height:48px;"><div style="position:absolute;inset:0;border-radius:50%;background:rgba(29,158,117,0.3);animation:pulse 1.8s ease-out infinite;"></div><div style="position:absolute;inset:8px;border-radius:50%;background:rgba(29,158,117,0.5);animation:pulse 1.8s ease-out 0.4s infinite;"></div><div style="position:absolute;inset:16px;border-radius:50%;background:#1d9e75;border:2px solid white;box-shadow:0 0 12px rgba(29,158,117,0.9);"></div></div><style>@keyframes pulse{{0%{{transform:scale(1);opacity:0.8;}}100%{{transform:scale(2.5);opacity:0;}}}}</style>',
  iconSize:[48,48], iconAnchor:[24,24], popupAnchor:[0,-28], className:''
}});

var dmpMarker = L.marker([11.5591, 43.1178], {{icon: pulseIcon}}).addTo(map);
dmpMarker.bindPopup('<div class="popup-title">{_popup_logo} Doraleh Multi-Purpose Port</div><div class="popup-sub">Port DMP / PDSA<br>Lat: 11.5591° N · Lon: 43.1178° E<br><span style="color:#4ade80;font-weight:600;"></span></div>', {{maxWidth:240}}).openPopup();

L.circle([11.5591, 43.1178], {{
  color:'rgba(29,158,117,0.6)', fillColor:'rgba(29,158,117,0.08)',
  fillOpacity:1, radius:400, weight:1.5
}}).addTo(map);

L.control.attribution({{prefix:'© CartoDB'}}).addTo(map);
</script></body></html>"""
        components.html(djibouti_map_html, height=610)
       

    if predict_btn:
        if pipe is None:
            st.error("Modèle non chargé. Vérifiez que meilleur_modele_foret_simple.pkl est présent.")
        else:
            # Poids_Camion_Entree = Poids_Tare (identique au Gradio)
            poids_entree = poids_tare
            data = pd.DataFrame([{
                'Poids_Tare_kg'             : poids_tare,
                'Poids_Cargaison_kg'        : poids_cargaison,
                'Poids_Camion_Entree_kg'    : poids_entree,
                'heure_entree'              : heure,
                'jour_semaine'              : jour,
                'mois'                      : mois,
                'log_Poids_Tare_kg'         : np.log1p(poids_tare),
                'log_Poids_Cargaison_kg'    : np.log1p(poids_cargaison),
                'log_Poids_Camion_Entree_kg': np.log1p(poids_entree),
                'Nom_Cargaison'             : nom_carg,
                'Nom_Navire'                : nom_nav,
                'Operateur_Entree'          : OP_ENTREE_DEFAULT,
                'Operateur_Sortie'          : OP_SORTIE_DEFAULT,
                'Type_Travail'              : type_trav,
            }])
            pred   = pipe.predict(data)[0]
            heures = int(pred // 60)
            mins   = int(pred % 60)
            risque_txt  = "Élevé" if pred > 1440 else "Modéré" if pred > 600 else "Faible"
            risk_cls    = "risk-high" if pred > 1440 else "risk-medium" if pred > 600 else "risk-low"
            risk_icon   = "fa-triangle-exclamation" if pred > 1440 else "fa-circle-exclamation" if pred > 600 else "fa-circle-check"
            needle_pct  = min(pred / 2880 * 100, 100)
            conseil     = "Surestarie critique — alerte immédiate recommandée au client." if pred > 1440 else \
                          "Attente significative — informer le client importateur." if pred > 600 else \
                          "Opération standard — durée d'attente normale."

            st.session_state.historique.append({
                "cargo": nom_carg, "heure": heure,
                "poids": poids_cargaison, "pred": pred, "risque": risque_txt
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
                color_p  = "#f87171" if poids_cargaison > 40000 else "#4ade80"
                alerte_cargo = "⚠" if nom_carg in ["Steel Bar","Steel Coil","Wheat","Rice"] else "✓"
                color_cargo  = "#f87171" if nom_carg in ["Steel Bar","Steel Coil","Wheat","Rice"] else "#4ade80"
                vs_moy  = pred - 768
                vs_sign = "+" if vs_moy >= 0 else ""
                vs_col  = "#f87171" if vs_moy > 0 else "#4ade80"

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
    st.markdown('<div class="glass-card-title" style="margin-bottom:8px;"><i class="fa-solid fa-clock-rotate-left"></i> &nbsp;Historique de la session</div>', unsafe_allow_html=True)
    if not st.session_state.historique:
        st.markdown('<p style="color:rgba(255,255,255,0.3);font-size:0.85rem;text-align:center;padding:20px;">Aucune prédiction effectuée</p>', unsafe_allow_html=True)
    else:
        rows = ""
        for h in reversed(st.session_state.historique[-10:]):
            tag_cls = "tag-low" if h["risque"]=="Faible" else "tag-medium" if h["risque"]=="Modéré" else "tag-high"
            rows += f"""<tr>
              <td>{h['cargo']}</td><td>{h['heure']}h</td>
              <td>{h['poids']:,.0f} kg</td>
              <td><b style="color:white;">{h['pred']:.0f} min</b></td>
              <td><span class="{tag_cls}">{h['risque']}</span></td>
            </tr>"""
        st.markdown(f"""
        <table class="hist-table">
          <thead><tr><th>Cargo</th><th>Heure</th><th>Poids cargo</th><th>Prédiction</th><th>Risque</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COMPARAISON
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "compare":

    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow"><i class="fa-solid fa-chart-bar"></i> Évaluation des algorithmes</div>
        <h1 class="page-header-title">Comparaison des modèles ML</h1>
        <p class="page-header-sub">Performance des 4 algorithmes sur le jeu de test indépendant — 3 187 observations jamais vues.</p>
    </div>
    """, unsafe_allow_html=True)

    max_r2   = max(m[1] for m in MODELES_DATA)
    max_mae  = max(m[2] for m in MODELES_DATA)
    max_rmse = max(m[3] for m in MODELES_DATA)

    # ── Animated metric-by-metric model cards ─────────────────────────────────
    models_json = []
    for nom, r2, mae, rmse, color in MODELES_DATA:
        best  = nom.endswith("★")
        models_json.append({
            "nom": nom, "r2": r2, "mae": mae, "rmse": rmse, "color": color,
            "best": best,
            "r2_pct":   round(r2 / max_r2 * 100),
            "mae_pct":  round((1 - mae  / max_mae)  * 100),
            "rmse_pct": round((1 - rmse / max_rmse) * 100),
        })

    import json
    animated_cards_html = f"""<!DOCTYPE html><html><head>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:transparent;font-family:'DM Sans',sans-serif;color:white;padding:8px 0;}}
.cards{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
.card{{
  background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:20px;padding:1.4rem;opacity:0;transform:translateY(18px);
  transition:border-color 0.3s,transform 0.3s;
}}
.card.visible{{opacity:1;transform:translateY(0);}}
.card.best{{border-color:rgba(29,158,117,0.4);background:linear-gradient(135deg,rgba(29,158,117,0.08),rgba(59,130,246,0.05));}}
.card:hover{{transform:translateY(-4px)!important;}}
.card-head{{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem;flex-wrap:wrap;gap:6px;}}
.card-name{{font-family:'Syne',sans-serif;font-size:0.98rem;font-weight:800;color:white;}}
.best-pill{{display:inline-flex;align-items:center;gap:5px;background:rgba(29,158,117,0.2);
  border:1px solid rgba(29,158,117,0.4);color:#4ade80;padding:3px 12px;border-radius:99px;
  font-size:0.68rem;font-weight:700;}}
.metric-block{{margin-bottom:0.9rem;}}
.metric-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;}}
.metric-label{{font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:rgba(255,255,255,0.35);}}
.metric-value{{font-size:0.82rem;font-weight:700;color:rgba(255,255,255,0.75);}}
.track{{height:7px;background:rgba(255,255,255,0.07);border-radius:99px;overflow:hidden;}}
.fill{{height:100%;border-radius:99px;width:0%;transition:width 0.85s cubic-bezier(0.22,1,0.36,1);}}
.metric-desc{{font-size:0.65rem;color:rgba(255,255,255,0.22);margin-top:3px;}}
</style></head><body>
<div class="cards" id="cards"></div>
<script>
var models={json.dumps(models_json, ensure_ascii=False)};
var container=document.getElementById('cards');

models.forEach(function(m,i){{
  var pill=m.best?'<span class="best-pill"><i class="fa-solid fa-trophy"></i> Modèle retenu</span>':'';
  var card=document.createElement('div');
  card.className='card'+(m.best?' best':'');
  card.style.transitionDelay=(i*0.1)+'s';
  card.innerHTML=`
    <div class="card-head">
      <span class="card-name">${{m.nom}}</span>
      ${{pill}}
    </div>
    <div class="metric-block" data-delay="0">
      <div class="metric-header">
        <span class="metric-label"><i class="fa-solid fa-chart-line"></i> R² — Score</span>
        <span class="metric-value">${{m.r2.toFixed(3)}}</span>
      </div>
      <div class="track"><div class="fill" data-pct="${{m.r2_pct}}" style="background:${{m.color}};"></div></div>
      <div class="metric-desc">Part de variance expliquée — plus proche de 1 est meilleur</div>
    </div>
    <div class="metric-block" data-delay="200">
      <div class="metric-header">
        <span class="metric-label"><i class="fa-solid fa-bullseye"></i> MAE — Précision</span>
        <span class="metric-value">${{m.mae}} min</span>
      </div>
      <div class="track"><div class="fill" data-pct="${{m.mae_pct}}" style="background:${{m.color}};opacity:0.75;"></div></div>
      <div class="metric-desc">Erreur absolue moyenne — plus bas est meilleur</div>
    </div>
    <div class="metric-block" data-delay="400">
      <div class="metric-header">
        <span class="metric-label"><i class="fa-solid fa-wave-square"></i> RMSE — Robustesse</span>
        <span class="metric-value">${{m.rmse}} min</span>
      </div>
      <div class="track"><div class="fill" data-pct="${{m.rmse_pct}}" style="background:${{m.color}};opacity:0.5;"></div></div>
      <div class="metric-desc">Erreur quadratique — pénalise les grands écarts</div>
    </div>
  `;
  container.appendChild(card);
  setTimeout(function(){{card.classList.add('visible');}}, 80+i*120);
}});

// Animate bars one by one with stagger
setTimeout(function(){{
  document.querySelectorAll('.metric-block').forEach(function(block){{
    var delay=parseInt(block.getAttribute('data-delay')||0);
    var fill=block.querySelector('.fill');
    var pct=fill.getAttribute('data-pct');
    setTimeout(function(){{fill.style.width=pct+'%';}}, 300+delay);
  }});
}}, 200);
</script></body></html>"""

    components.html(animated_cards_html, height=560)

    # ── Histogramme interactif Plotly — comparaison des modèles ──────────────
    st.markdown('<div class="glass-card" style="margin-top:1rem;"><div class="glass-card-title"><i class="fa-solid fa-chart-column"></i> Histogramme interactif — métriques par modèle</div>', unsafe_allow_html=True)

    noms_modeles  = [m[0].replace(" ★", "") for m in MODELES_DATA]
    r2_vals       = [m[1] for m in MODELES_DATA]
    mae_vals      = [m[2] for m in MODELES_DATA]
    rmse_vals     = [m[3] for m in MODELES_DATA]
    couleurs      = [m[4] for m in MODELES_DATA]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        name="R² (score)", x=noms_modeles, y=r2_vals,
        marker_color="#1d9e75",
        hovertemplate="<b>%{x}</b><br>R² = %{y:.3f}<extra></extra>",
        yaxis="y1"
    ))
    fig_hist.add_trace(go.Bar(
        name="MAE (min)", x=noms_modeles, y=mae_vals,
        marker_color="#22d3ee",
        hovertemplate="<b>%{x}</b><br>MAE = %{y} min<extra></extra>",
        yaxis="y2"
    ))
    fig_hist.add_trace(go.Bar(
        name="RMSE (min)", x=noms_modeles, y=rmse_vals,
        marker_color="#3b82f6",
        hovertemplate="<b>%{x}</b><br>RMSE = %{y} min<extra></extra>",
        yaxis="y2"
    ))

    fig_hist.update_layout(
        barmode="group",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.65)", family="DM Sans, sans-serif", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, color="rgba(255,255,255,0.55)"),
        yaxis=dict(title="R²", side="left", range=[0, 1],
                   showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="rgba(255,255,255,0.45)"),
        yaxis2=dict(title="Minutes (MAE / RMSE)", side="right", overlaying="y",
                    showgrid=False, color="rgba(255,255,255,0.45)"),
        hoverlabel=dict(bgcolor="#0a0f1a", font_color="white", bordercolor="#1d9e75"),
        bargap=0.25, bargroupgap=0.12,
    )

    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<p style="font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:8px;">Survolez les barres pour le détail. Échelle gauche = R² · échelle droite = minutes (MAE/RMSE).</p></div>', unsafe_allow_html=True)

   

    radar_html = """<!DOCTYPE html><html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>*{margin:0;padding:0;box-sizing:border-box;}body{background:transparent;}canvas{max-width:100%;}</style>
</head><body>
<canvas id="radar" height="280"></canvas>
<script>
new Chart(document.getElementById('radar'),{
  type:'radar',
  data:{
    labels:['R² (perf)','MAE (précision)','RMSE (robustesse)','Généralisation','Interprétabilité'],
    datasets:[
      {label:'Forêt Aléatoire',data:[100,100,100,85,60],borderColor:'#22c55e',backgroundColor:'rgba(34,197,94,0.12)',pointBackgroundColor:'#22c55e',borderWidth:2},
      {label:'KNN',data:[88,82,87,80,50],borderColor:'#f97316',backgroundColor:'rgba(249,115,22,0.08)',pointBackgroundColor:'#f97316',borderWidth:2},
      {label:'Arbre de Décision',data:[73,58,68,60,95],borderColor:'#eab308',backgroundColor:'rgba(234,179,8,0.08)',pointBackgroundColor:'#eab308',borderWidth:2},
      {label:'Régression Linéaire',data:[24,0,0,50,100],borderColor:'#f43f5e',backgroundColor:'rgba(244,63,94,0.08)',pointBackgroundColor:'#f43f5e',borderWidth:2},
    ]
  },
  options:{
    responsive:true,
    plugins:{legend:{labels:{color:'rgba(255,255,255,0.6)',font:{size:11}}}},
    scales:{r:{grid:{color:'rgba(255,255,255,0.07)'},pointLabels:{color:'rgba(255,255,255,0.5)',font:{size:10}},ticks:{display:false},angleLines:{color:'rgba(255,255,255,0.06)'}}}
  }
});
</script></body></html>"""

    st.markdown('<div class="glass-card" style="margin-top:1rem;"><div class="glass-card-title"><i class="fa-solid fa-chart-radar"></i> Radar — performance globale</div>', unsafe_allow_html=True)
    components.html(radar_html, height=320)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "about":

    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow"><i class="fa-solid fa-circle-info"></i> À propos du projet</div>
        <h1 class="grad">Port Intelligence </h1>
     
    </div>
    """, unsafe_allow_html=True)

    # ── Contexte avec photo à droite ──────────────────────────────────────────
    _port1_path = "assets/port_1.jpg"
    if not os.path.exists(_port1_path):
        _port1_path = "assets/port_1.png"
    _port1_uri = img_to_b64(_port1_path) if os.path.exists(_port1_path) else None

    _col_txt, _col_img = st.columns([1.3, 1])

    with _col_txt:
        st.markdown("""
        <div class="about-eyebrow"><i class="fa-solid fa-anchor"></i> Contexte</div>
        <h2 class="about-section-title">Optimiser les opérations du <span class="grad">Doraleh Multipurpose Port</span></h2>
        <p class="about-sub">
            La surestarie — durée d'attente d'un camion entre la Gate In et la Gate Out — représente
            un coût financier direct pour les importateurs. Avec une moyenne de 768 minutes par camion,
            anticiper ces délais permet au port de mieux planifier ses ressources et aux clients
            d'optimiser leur logistique. Ce projet est le premier à appliquer le Machine Learning
            sur des données opérationnelles réelles et confidentielles du DMP.
        </p>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:1.2rem;">
          <span style="display:inline-flex;align-items:center;gap:7px;background:rgba(29,158,117,0.1);border:1px solid rgba(29,158,117,0.25);color:#4ade80;padding:5px 14px;border-radius:99px;font-size:0.78rem;font-weight:600;"><i class="fa-solid fa-database" style="font-size:0.68rem;"></i> 62 329 enregistrements réels</span>
          <span style="display:inline-flex;align-items:center;gap:7px;background:rgba(29,158,117,0.1);border:1px solid rgba(29,158,117,0.25);color:#4ade80;padding:5px 14px;border-radius:99px;font-size:0.78rem;font-weight:600;"><i class="fa-solid fa-globe" style="font-size:0.68rem;"></i> Déployé sur Hugging Face</span>
          <span style="display:inline-flex;align-items:center;gap:7px;background:rgba(29,158,117,0.1);border:1px solid rgba(29,158,117,0.25);color:#4ade80;padding:5px 14px;border-radius:99px;font-size:0.78rem;font-weight:600;"><i class="fa-solid fa-shield" style="font-size:0.68rem;"></i> Données confidentielles DMP</span>
        </div>
        """, unsafe_allow_html=True)

    with _col_img:
        if _port1_uri:
            st.markdown(
                '<div style="border-radius:20px;overflow:hidden;border:1px solid rgba(29,158,117,0.25);'
                'box-shadow:0 20px 60px rgba(0,0,0,0.5);position:relative;">'
                f'<img src="{_port1_uri}" style="width:100%;display:block;object-fit:cover;max-height:340px;" />'
                '<div style="position:absolute;bottom:0;left:0;right:0;'
                'background:linear-gradient(transparent,rgba(6,9,15,0.85));padding:1rem 1.2rem;">'
                '<div style="font-size:0.72rem;font-weight:600;color:#4ade80;letter-spacing:0.08em;text-transform:uppercase;">'
                '<i class="fa-solid fa-anchor"></i> Doraleh Multi-Purpose Port</div>'
                '<div style="font-size:0.65rem;color:rgba(255,255,255,0.4);margin-top:2px;"></div>'
                '</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="border-radius:20px;border:1px solid rgba(29,158,117,0.2);'
                'background:rgba(29,158,117,0.06);height:280px;'
                'display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.7rem;">'
                '<i class="fa-solid fa-image" style="font-size:2.5rem;color:rgba(29,158,117,0.4);"></i>'
                '<p style="font-size:0.78rem;color:rgba(255,255,255,0.3);text-align:center;max-width:180px;">'
                'Placez <b style="color:#4ade80;">assets/port_1.jpg</b> pour afficher la photo du port'
                '</p></div>',
                unsafe_allow_html=True
            )

    st.markdown('<hr class="a-divider">', unsafe_allow_html=True)

  

    

    st.markdown("""
    <div class="about-eyebrow"><i class="fa-solid fa-chart-column"></i> Exploration des données</div>
    <h2 class="about-section-title">Statistiques <span class="grad">clés du port</span></h2>
    """, unsafe_allow_html=True)

    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        st.markdown('<div class="glass-card"><div class="glass-card-title"><i class="fa-solid fa-clock"></i> Volume camions / heure d\'entrée (top 12)</div>', unsafe_allow_html=True)

        heures_lbl = [h for h, _ in DATA_HEURES]
        heures_val = [v for _, v in DATA_HEURES]
        heures_col = ["#1d9e75" if h in ("16h", "17h", "15h") else "rgba(255,255,255,0.25)" for h in heures_lbl]

        fig_h = go.Figure(go.Bar(
            x=heures_val, y=heures_lbl, orientation="h",
            marker_color=heures_col,
            hovertemplate="<b>%{y}</b><br>%{x:,} camions<extra></extra>",
        ))
        fig_h.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.6)", family="DM Sans, sans-serif", size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="rgba(255,255,255,0.4)"),
            yaxis=dict(autorange="reversed", showgrid=False, color="rgba(255,255,255,0.5)"),
            hoverlabel=dict(bgcolor="#0a0f1a", font_color="white", bordercolor="#1d9e75"),
            bargap=0.25,
        )
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})
        st.markdown('<p style="font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:8px;">Pic à 16h : 2 700 camions — fin des opérations administratives</p></div>', unsafe_allow_html=True)

    with col_eda2:
        st.markdown('<div class="glass-card"><div class="glass-card-title"><i class="fa-solid fa-box"></i> Surestarie moyenne par cargo (min)</div>', unsafe_allow_html=True)

        cargo_lbl = [lbl for lbl, _, _ in DATA_CARGO]
        cargo_val = [v for _, v, _ in DATA_CARGO]
        cargo_col = [color for _, _, color in DATA_CARGO]

        fig_c = go.Figure(go.Bar(
            x=cargo_val, y=cargo_lbl, orientation="h",
            marker_color=cargo_col,
            hovertemplate="<b>%{y}</b><br>%{x:,} min<extra></extra>",
        ))
        fig_c.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.6)", family="DM Sans, sans-serif", size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="rgba(255,255,255,0.4)"),
            yaxis=dict(autorange="reversed", showgrid=False, color="rgba(255,255,255,0.5)"),
            hoverlabel=dict(bgcolor="#0a0f1a", font_color="white", bordercolor="#1d9e75"),
            bargap=0.25,
        )
        st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})
        st.markdown('<p style="font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:8px;">Steel : 4 500 min (~75h) — manutention complexe et contraintes sécurité</p></div>', unsafe_allow_html=True)

    st.markdown('<hr class="a-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="about-eyebrow"><i class="fa-solid fa-boxes-stacked"></i> Algorithmes</div>
    <h2 class="about-section-title">Deux meilleurs <span class="grad">modèles</span></h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="arch-duo">
      <div class="arch-card rf">
        <div class="arch-placeholder" style="background:linear-gradient(135deg,rgba(29,158,117,0.15),rgba(59,130,246,0.1));">
          <i class="fa-solid fa-tree" style="color:#1d9e75;"></i>
          <p>Modèle retenu — Forêt Aléatoire</p>
        </div>
        <div class="arch-body">
          <div class="arch-badge badge-rf"><i class="fa-solid fa-trophy"></i> Modèle retenu</div>
          <div class="arch-title">Forêt Aléatoire (Random Forest)</div>
          <p class="arch-desc">Ensemble de 100 arbres de décision entraînés sur des sous-ensembles aléatoires — technique de bagging. Réduit l'overfitting tout en capturant les relations non-linéaires complexes.</p>
          <div class="arch-spec"><i class="fa-solid fa-check" style="color:#1d9e75;"></i> R²=0.667 (test) — MAE=260 min</div>
          <div class="arch-spec"><i class="fa-solid fa-check" style="color:#1d9e75;"></i> Imputation simple (médiane/mode)</div>
          <div class="arch-spec"><i class="fa-solid fa-check" style="color:#1d9e75;"></i> Feature Importance — heure_entrée top 1</div>
          <div class="arch-spec"><i class="fa-solid fa-check" style="color:#1d9e75;"></i> Correction data leakage Poids_Sortie</div>
        </div>
      </div>
      <div class="arch-card knn">
        <div class="arch-placeholder" style="background:linear-gradient(135deg,rgba(249,115,22,0.15),rgba(234,179,8,0.1));">
          <i class="fa-solid fa-route" style="color:#f97316;"></i>
          <p>Alternative — KNN optimisé</p>
        </div>
        <div class="arch-body">
          <div class="arch-badge badge-knn"><i class="fa-solid fa-medal"></i> Alternative crédible</div>
          <div class="arch-title">K-Nearest Neighbors (KNN)</div>
          <p class="arch-desc">Prédit la surestarie d'une nouvelle opération en se basant sur la moyenne des K opérations les plus similaires. Optimisé par GridSearchCV.</p>
          <div class="arch-spec"><i class="fa-solid fa-check" style="color:#f97316;"></i> R²=0.587 (validation) — MAE=246 min</div>
          <div class="arch-spec"><i class="fa-solid fa-check" style="color:#f97316;"></i> Imputation KNN (k voisins)</div>
          <div class="arch-spec"><i class="fa-solid fa-check" style="color:#f97316;"></i> StandardScaler obligatoire</div>
          <div class="arch-spec"><i class="fa-solid fa-check" style="color:#f97316;"></i> Temps d'entraînement plus rapide</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="a-divider">', unsafe_allow_html=True)

    techs = [
        ("Python",      "Langage principal", "#3799d7", "rgba(55,153,215,0.1)", "rgba(55,153,215,0.2)", "#3799d7",
         '<i class="fa-brands fa-python" style="font-size:24px;color:#3799d7;"></i>'),
        ("Scikit-learn","ML & Pipelines",    "#f89820", "rgba(248,152,32,0.1)",  "rgba(248,152,32,0.2)",  "#f89820",
         '<i class="fa-solid fa-gears" style="font-size:22px;color:#f89820;"></i>'),
        ("Pandas",      "Données tabulaires","#130754", "rgba(29,158,117,0.1)", "rgba(29,158,117,0.2)", "#1d9e75",
         '<i class="fa-solid fa-table" style="font-size:22px;color:#1d9e75;"></i>'),
        ("Streamlit",   "Interface web",     "#ff4b4b", "rgba(255,75,75,0.1)",  "rgba(255,75,75,0.2)",  "#ff4b4b",
         '<i class="fa-solid fa-bolt" style="font-size:22px;color:#ff4b4b;"></i>'),
        ("Power BI",    "Dashboard",         "#f2c811", "rgba(242,200,17,0.1)", "rgba(242,200,17,0.2)", "#f2c811",
         '<i class="fa-solid fa-chart-pie" style="font-size:22px;color:#f2c811;"></i>'),
        ("Gradio",      "API prédiction",    "#f97316", "rgba(249,115,22,0.1)", "rgba(249,115,22,0.2)", "#f97316",
         '<i class="fa-solid fa-sliders" style="font-size:22px;color:#f97316;"></i>'),
    ]

    st.markdown("""
    <div class="about-eyebrow"><i class="fa-solid fa-code"></i> Stack technique</div>
    <h2 class="about-section-title">Technologies <span class="grad">utilisées</span></h2>
    <p class="about-sub">Un ensemble d'outils Python modernes pour le Machine Learning, la visualisation et le déploiement.</p>
    """, unsafe_allow_html=True)

    tech_cards_html = ""
    for name, role, color, bg, border, bar, icon_html in techs:
        tech_cards_html += f"""
<div class="tech-card">
  <div class="tech-icon" style="background:{bg};border:1px solid {border};">{icon_html}</div>
  <div class="tech-name">{name}</div>
  <div class="tech-role">{role}</div>
  <div class="tech-bar" style="background:{bar};"></div>
</div>"""

    st.markdown(f'<div class="tech-6">{tech_cards_html}</div>', unsafe_allow_html=True)

    st.markdown("""
    <script>
    (function(){
      var io=new IntersectionObserver(function(entries){
        entries.forEach(function(e){
          if(e.isIntersecting){e.target.style.opacity='1';e.target.style.transform='translateY(0)';io.unobserve(e.target);}
        });
      },{threshold:0.1});
      document.querySelectorAll('.model-card,.arch-card,.tech-card,.pipeline-step').forEach(function(el,i){
        el.style.opacity='0';el.style.transform='translateY(22px)';
        el.style.transition='opacity 0.5s ease '+i*0.04+'s, transform 0.5s cubic-bezier(0.22,1,0.36,1) '+i*0.04+'s';
        io.observe(el);
      });
    })();
    </script>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
def get_footer_logo():
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        uri = img_to_b64(logo_path)
        return f'<img src="{uri}" style="width:100%;height:100%;object-fit:contain;padding:4px;border-radius:8px;" />'
    return '<i class="fa-solid fa-anchor"></i>'

_footer_logo = get_footer_logo()
st.markdown(f"""
<div class="site-footer">
  <div class="footer-grid">
    <div>
      <div class="footer-brand-row">
        <div class="footer-logo-box">{_footer_logo}</div>
        <span class="footer-app-name">Port<span>Intelligence</span> </span>
      </div>
      <p class="footer-desc">Prédiction de la surestarie par Machine Learning sur données réelles du Doraleh Multipurpose Port — Djibouti.</p>
    </div>
    <div>
      <div class="footer-col-title">Navigation</div>
      <a class="footer-link" href="?page=home" target="_self"><i class="fa-solid fa-anchor"></i> Accueil</a>
      <a class="footer-link" href="?page=predict" target="_self"><i class="fa-solid fa-clock"></i> Prédiction</a>
      <a class="footer-link" href="?page=compare" target="_self"><i class="fa-solid fa-chart-bar"></i> Comparaison</a>
      <a class="footer-link" href="?page=about" target="_self"><i class="fa-solid fa-circle-info"></i> À propos</a>
    </div>
    <div>
      <div class="footer-col-title">Projet</div>
      <a class="footer-link" href="https://github.com/HASSAN-MAHAMOUD/port-intelligence-ai" target="_blank"><i class="fa-brands fa-github"></i> Code source — GitHub</a>
      <a class="footer-link" href="https://huggingface.co/spaces/HassanML2026/port-intelligence-ai" target="_blank"><i class="fa-solid fa-rocket"></i> Démo Hugging Face</a>
    </div>
  </div>
  <div class="footer-bottom">
    <div class="footer-copy"> PortIntelligence © 2026 — Tous droits réservés</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)