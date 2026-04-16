#!/usr/bin/env python3
"""
Admin Panel — Poverty Stoplight RAG
Gestión de documentos, chunks, modelos y embeddings.
"""

import os
import sys
import hashlib
import secrets
import subprocess
from pathlib import Path

import psycopg2
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuración de rutas y entorno
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "db_rag"),
    user=os.getenv("DB_USER", "sgadmin"),
    password=os.getenv("DB_PASSWORD", "sg4dm1n!"),
)

EMBEDDING_TABLES = [
    ("bge-m3",                 "embeddings_bge_m3",    1024),
    ("nomic-embed-text",       "embeddings_nomic",      768),
    ("mxbai-embed-large",      "embeddings_mxbai",     1024),
    ("all-minilm",             "embeddings_minilm",     384),
    ("snowflake-arctic-embed", "embeddings_snowflake",  1024),
]

# ---------------------------------------------------------------------------
# Helpers de base de datos
# ---------------------------------------------------------------------------

def new_conn():
    return psycopg2.connect(**DB_CONFIG)


def query_df(sql: str, params=None) -> pd.DataFrame:
    conn = new_conn()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def fetchone(sql: str, params=None):
    conn = new_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()
    finally:
        conn.close()


def execute_sql(sql: str, params=None):
    conn = new_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def render_table(df: pd.DataFrame) -> None:
    """Renderiza un DataFrame como tabla HTML con estilo industrial."""
    if df is None or df.empty:
        st.markdown('<div class="ind-empty">Sin datos para mostrar.</div>', unsafe_allow_html=True)
        return

    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    rows = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{'' if pd.isna(v) else v}</td>" for v in row.values)
        rows += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <div class="ind-table-wrap">
        <table class="ind-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def db_ok() -> bool:
    try:
        fetchone("SELECT 1")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def hash_password(password: str) -> tuple:
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return hashed, salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return secrets.compare_digest(check, hashed)


def ensure_users_table():
    """Create admin_users table if it doesn't exist and seed a default admin."""
    execute_sql("""
        CREATE TABLE IF NOT EXISTS admin_users (
            id           SERIAL PRIMARY KEY,
            username     VARCHAR(80)  NOT NULL UNIQUE,
            email        VARCHAR(200),
            password_hash VARCHAR(64) NOT NULL,
            salt         VARCHAR(32)  NOT NULL,
            role         VARCHAR(20)  NOT NULL DEFAULT 'viewer',
            is_active    BOOLEAN      NOT NULL DEFAULT TRUE,
            created_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Seed default admin only if table is empty
    count = fetchone("SELECT COUNT(*) FROM admin_users")[0]
    if count == 0:
        ph, salt = hash_password("admin1234")
        execute_sql(
            "INSERT INTO admin_users (username, email, password_hash, salt, role) VALUES (%s, %s, %s, %s, 'admin')",
            ("admin", "admin@localhost", ph, salt),
        )


def authenticate_user(username: str, password: str):
    """Return user dict on success, None on failure."""
    conn = new_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, email, password_hash, salt, role, is_active "
                "FROM admin_users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None
    uid, uname, email, ph, salt, role, active = row
    if not active:
        return None
    if not verify_password(password, ph, salt):
        return None
    return {"id": uid, "username": uname, "email": email, "role": role}


# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Admin | Poverty Stoplight RAG",
    page_icon="⬡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS INDUSTRIAL — REDISEÑO TOTAL
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">

<style>
/* ═══════════════════════════════════════════════════════════════════════════
   VARIABLES INDUSTRIALES
═══════════════════════════════════════════════════════════════════════════ */
:root {
    --bg-base:        #0D7A34;
    --bg-deep:        #052012;
    --bg-panel:       #083D1C;
    --bg-card:        #0A5224;
    --bg-hover:       #0E6B2E;
    --accent-neon:    #00FF85;
    --accent-amber:   #FFD600;
    --accent-red:     #FF3B3B;
    --accent-ice:     #00DFFF;
    --text-primary:   #F0FFF4;
    --text-secondary: #86EFAC;
    --text-muted:     #4ADE80;
    --text-data:      #D1FAE5;
    --border-neon:    rgba(0, 255, 133, 0.2);
    --border-amber:   rgba(255, 214, 0, 0.4);
    --border-subtle:  rgba(255, 255, 255, 0.06);
    --shadow-deep:    0 8px 40px rgba(0, 0, 0, 0.6);
    --shadow-neon:    0 0 24px rgba(0, 255, 133, 0.2);
    --glow-amber:     0 0 20px rgba(255, 214, 0, 0.35);
    --radius-sharp:   2px;
    --radius-card:    4px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   KEYFRAMES — ANIMACIONES
═══════════════════════════════════════════════════════════════════════════ */
@keyframes pulse-neon {
    0%, 100% { box-shadow: 0 0 6px rgba(0,255,133,0.6), 0 0 12px rgba(0,255,133,0.3); opacity: 1; }
    50%       { box-shadow: 0 0 14px rgba(0,255,133,1),   0 0 28px rgba(0,255,133,0.6); opacity: 0.7; }
}
@keyframes pulse-dot {
    0%, 100% { transform: scale(1);   opacity: 1; }
    50%       { transform: scale(1.4); opacity: 0.6; }
}
@keyframes slideInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes shimmer-bar {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}
@keyframes grid-drift {
    0%   { background-position: 0 0; }
    100% { background-position: 40px 40px; }
}
@keyframes border-glow {
    0%, 100% { border-color: rgba(0,255,133,0.2); }
    50%       { border-color: rgba(0,255,133,0.6); }
}
@keyframes counter-in {
    from { opacity: 0; transform: scale(0.7) translateY(8px); }
    to   { opacity: 1; transform: scale(1)   translateY(0); }
}

/* ═══════════════════════════════════════════════════════════════════════════
   BASE — FONDO CON CUADRÍCULA INDUSTRIAL
═══════════════════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif !important;
}

.stApp {
    background-color: var(--bg-base) !important;
    background-image:
        linear-gradient(rgba(0,255,133,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,133,0.04) 1px, transparent 1px),
        radial-gradient(ellipse at 0% 0%, rgba(0,255,133,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(255,214,0,0.05) 0%, transparent 50%);
    background-size: 40px 40px, 40px 40px, 100% 100%, 100% 100%;
    animation: grid-drift 20s linear infinite;
}


.block-container {
    background: transparent !important;
    padding-top: 1.5rem !important;
    max-width: 1400px !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   TIPOGRAFÍA
═══════════════════════════════════════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6, p, span, div, label, input,
textarea, select, button, [class*="css"] {
    font-family: 'Montserrat', sans-serif !important;
}

h1 {
    color: var(--text-primary) !important;
    font-weight: 900 !important;
    font-size: 1.8rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

h2 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 1.1rem !important;
    border-left: 3px solid var(--accent-neon) !important;
    padding-left: 12px !important;
    margin-bottom: 1rem !important;
}

h3 {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    font-size: 0.9rem !important;
}

body, .stApp, .stMarkdown p, label,
[data-testid="stCaptionContainer"], .stCaption {
    color: var(--text-data) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   DIVIDER
═══════════════════════════════════════════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,255,133,0.15) !important;
    margin: 1.5rem 0 !important;
    position: relative;
}
hr::after {
    content: '◆';
    position: absolute;
    left: 50%;
    transform: translateX(-50%) translateY(-55%);
    color: var(--accent-neon);
    font-size: 0.55rem;
    background: var(--bg-base);
    padding: 0 6px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR / HEADER OCULTO DE STREAMLIT
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebarNav"], #MainMenu, footer { display: none !important; }
header[data-testid="stHeader"] {
    background: rgba(5,32,18,0.85) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-neon) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   TABS — INDUSTRIAL
═══════════════════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-deep) !important;
    border-bottom: 1px solid var(--border-neon) !important;
    gap: 0 !important;
    padding: 0 !important;
    border-radius: 0 !important;
    animation: slideInDown 0.4s ease both;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 0 !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 14px 22px !important;
    border: none !important;
    border-right: 1px solid rgba(0,255,133,0.08) !important;
    transition: all 0.2s ease !important;
    position: relative !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(0,255,133,0.05) !important;
    color: var(--accent-neon) !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(0,255,133,0.08) !important;
    color: var(--accent-neon) !important;
    font-weight: 800 !important;
    border-bottom: 2px solid var(--accent-neon) !important;
}

.stTabs [aria-selected="true"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-neon);
    box-shadow: var(--shadow-neon);
}

.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1.8rem !important;
    animation: slideInUp 0.35s ease both;
}

/* ═══════════════════════════════════════════════════════════════════════════
   MÉTRICAS — TARJETAS INDUSTRIALES
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-neon) !important;
    border-top: 2px solid var(--accent-neon) !important;
    border-radius: var(--radius-card) !important;
    padding: 20px 24px !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    animation: counter-in 0.5s ease both;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: var(--shadow-neon), var(--shadow-deep) !important;
    border-color: rgba(0,255,133,0.5) !important;
}

/* Decoración angular en la esquina */
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 0; height: 0;
    border-style: solid;
    border-width: 0 28px 28px 0;
    border-color: transparent rgba(0,255,133,0.4) transparent transparent;
}

[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.02em !important;
    line-height: 1 !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: var(--accent-neon) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   DATAFRAMES
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-card) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-neon) !important;
    animation: fadeIn 0.4s ease both;
}

/* Radio buttons como selector de documentos */
.doc-radio [data-testid="stRadio"] > label {
    display: none;
}
.doc-radio [data-testid="stRadio"] > div {
    gap: 0 !important;
}
.doc-radio [data-testid="stRadio"] > div > label {
    background: rgba(0,255,133,0.04) !important;
    border: 1px solid var(--border-neon) !important;
    border-radius: 2px !important;
    padding: 8px 14px !important;
    margin-bottom: 3px !important;
    color: var(--text-data) !important;
    font-size: 0.78rem !important;
    font-family: 'Share Tech Mono', monospace !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.doc-radio [data-testid="stRadio"] > div > label:hover {
    background: rgba(0,255,133,0.10) !important;
    border-color: var(--accent-neon) !important;
    color: var(--accent-neon) !important;
}
.doc-radio [data-testid="stRadio"] > div > label[data-baseweb="radio"] input:checked + div,
.doc-radio [data-testid="stRadio"] > div > label[aria-checked="true"] {
    background: rgba(0,255,133,0.14) !important;
    border-color: var(--accent-neon) !important;
    color: var(--accent-neon) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   BOTONES — ESTILO INDUSTRIAL
═══════════════════════════════════════════════════════════════════════════ */
.stButton > button {
    background: transparent !important;
    color: var(--accent-neon) !important;
    border: 1px solid var(--accent-neon) !important;
    border-radius: var(--radius-sharp) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 10px 22px !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
    clip-path: polygon(8px 0%, 100% 0%, calc(100% - 8px) 100%, 0% 100%);
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,255,133,0.15), transparent);
    transition: left 0.3s ease;
}

.stButton > button:hover {
    background: rgba(0,255,133,0.1) !important;
    box-shadow: var(--shadow-neon) !important;
    transform: translateY(-2px) !important;
    border-color: var(--accent-neon) !important;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button[kind="primary"] {
    background: var(--accent-amber) !important;
    color: var(--bg-deep) !important;
    border-color: var(--accent-amber) !important;
    font-weight: 800 !important;
}

.stButton > button[kind="primary"]:hover {
    background: #FFE033 !important;
    box-shadow: var(--glow-amber) !important;
}

.stButton > button[kind="secondary"] {
    border-color: rgba(255,255,255,0.2) !important;
    color: var(--text-secondary) !important;
}

.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent-neon) !important;
    color: var(--accent-neon) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   INPUTS, SELECTBOX, TEXTAREA
═══════════════════════════════════════════════════════════════════════════ */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-panel) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(0,255,133,0.25) !important;
    border-radius: var(--radius-sharp) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-neon) !important;
    box-shadow: 0 0 0 2px rgba(0,255,133,0.15), var(--shadow-neon) !important;
}

/* Labels de inputs */
.stTextInput label, .stNumberInput label, .stTextArea label,
.stSelectbox label, .stCheckbox label, .stSlider label {
    color: var(--text-muted) !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
}

.stSelectbox > div > div,
[data-baseweb="select"] > div {
    background: var(--bg-panel) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(0,255,133,0.25) !important;
    border-radius: var(--radius-sharp) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

[data-baseweb="select"] > div:hover {
    border-color: var(--accent-neon) !important;
}

[data-baseweb="popover"] {
    background: var(--bg-deep) !important;
    border: 1px solid var(--border-neon) !important;
    border-radius: var(--radius-card) !important;
}

[data-baseweb="option"] {
    background: transparent !important;
    color: var(--text-data) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: background 0.15s ease !important;
}

[data-baseweb="option"]:hover {
    background: rgba(0,255,133,0.08) !important;
    color: var(--accent-neon) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   PROGRESS BARS — INDUSTRIAL STRIPED
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stProgressBar"] > div {
    background: var(--bg-panel) !important;
    border-radius: 0 !important;
    height: 10px !important;
    border: 1px solid rgba(0,255,133,0.15) !important;
    overflow: hidden;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(
        90deg,
        #00FF85 0%,
        #00CC6A 40%,
        #00FF85 60%,
        #FFD600 100%
    ) !important;
    background-size: 200% auto !important;
    border-radius: 0 !important;
    animation: shimmer-bar 2.5s linear infinite !important;
    box-shadow: 0 0 8px rgba(0,255,133,0.5) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   ALERTS — INDUSTRIAL
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: var(--radius-sharp) !important;
    border-left-width: 4px !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.02em !important;
}

.stSuccess  {
    background: rgba(0, 255, 133, 0.08) !important;
    border-color: var(--accent-neon) !important;
    color: var(--text-primary) !important;
}
.stWarning  {
    background: rgba(255, 214, 0, 0.08) !important;
    border-color: var(--accent-amber) !important;
    color: var(--text-primary) !important;
}
.stError    {
    background: rgba(255, 59, 59, 0.08) !important;
    border-color: var(--accent-red) !important;
    color: var(--text-primary) !important;
}
.stInfo     {
    background: rgba(0, 223, 255, 0.08) !important;
    border-color: var(--accent-ice) !important;
    color: var(--text-primary) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   FILE UPLOADER
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--bg-panel) !important;
    border: 2px dashed rgba(0,255,133,0.3) !important;
    border-radius: var(--radius-card) !important;
    padding: 20px !important;
    transition: all 0.25s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-neon) !important;
    background: rgba(0,255,133,0.04) !important;
    box-shadow: var(--shadow-neon) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   CHECKBOX
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stCheckbox"] label {
    color: var(--text-data) !important;
    font-weight: 500 !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SPINNER
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSpinner"] {
    color: var(--accent-neon) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SCROLLBAR — INDUSTRIAL THIN
═══════════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar          { width: 4px; height: 4px; }
::-webkit-scrollbar-track    { background: var(--bg-deep); }
::-webkit-scrollbar-thumb    { background: var(--accent-neon); }
::-webkit-scrollbar-thumb:hover { background: #00CC6A; }

/* ═══════════════════════════════════════════════════════════════════════════
   SLIDER
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSlider"] > div > div > div {
    background: var(--accent-neon) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   COMPONENTE PERSONALIZADO: SECTION HEADER
═══════════════════════════════════════════════════════════════════════════ */
.ind-section {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.6rem 0 1rem;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(0,255,133,0.12);
}
.ind-section .bar {
    width: 4px;
    height: 22px;
    background: var(--accent-neon);
    box-shadow: 0 0 8px rgba(0,255,133,0.6);
    flex-shrink: 0;
}
.ind-section .label {
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent-neon);
}
.ind-section .count {
    margin-left: auto;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    font-family: 'Share Tech Mono', monospace !important;
}

/* ─── Model coverage card ─────────────────────────────────────────── */
.model-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-neon);
    border-left: 3px solid var(--accent-neon);
    padding: 12px 16px;
    margin-bottom: 8px;
    border-radius: var(--radius-card);
    display: flex;
    align-items: center;
    gap: 16px;
    transition: all 0.2s ease;
    animation: slideInUp 0.3s ease both;
}
.model-card:hover {
    background: var(--bg-hover);
    box-shadow: var(--shadow-neon);
}
.model-name {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-primary);
    width: 180px;
    flex-shrink: 0;
}
.model-dims {
    font-size: 0.65rem;
    font-family: 'Share Tech Mono', monospace;
    color: var(--accent-ice);
    width: 55px;
    flex-shrink: 0;
}
.model-bar-wrap {
    flex: 1;
    height: 6px;
    background: rgba(0,255,133,0.1);
    border-radius: 0;
    overflow: hidden;
}
.model-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #00FF85, #FFD600);
    box-shadow: 0 0 6px rgba(0,255,133,0.4);
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
    animation: shimmer-bar 3s linear infinite;
    background-size: 200% auto;
}
.model-pct {
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'Share Tech Mono', monospace;
    color: var(--accent-neon);
    width: 48px;
    text-align: right;
    flex-shrink: 0;
}
.model-count {
    font-size: 0.65rem;
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-muted);
    width: 110px;
    text-align: right;
    flex-shrink: 0;
}

/* ─── Status badge ───────────────────────────────────────────────── */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,255,133,0.1);
    border: 1px solid rgba(0,255,133,0.35);
    border-radius: 2px;
    padding: 5px 14px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent-neon);
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent-neon);
    animation: pulse-dot 1.8s ease infinite;
    box-shadow: 0 0 6px rgba(0,255,133,0.8);
    flex-shrink: 0;
}
.status-offline {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,59,59,0.1);
    border: 1px solid rgba(255,59,59,0.35);
    border-radius: 2px;
    padding: 5px 14px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent-red);
}
.status-dot-red {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent-red);
    animation: pulse-dot 0.8s ease infinite;
    flex-shrink: 0;
}

/* ─── Hero header ────────────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, var(--bg-deep) 0%, var(--bg-panel) 100%);
    border: 1px solid var(--border-neon);
    border-top: 3px solid var(--accent-neon);
    padding: 28px 36px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    animation: slideInDown 0.5s ease both;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,255,133,0.08) 0%, transparent 70%);
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--accent-neon), transparent, var(--accent-amber), transparent);
}
.hero-title {
    font-size: 1.5rem;
    font-weight: 900;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-primary);
    margin: 0;
    line-height: 1;
}
.hero-sub {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent-neon);
    margin: 6px 0 0;
    opacity: 0.85;
}
.hero-id {
    position: absolute;
    top: 16px;
    right: 36px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: rgba(0,255,133,0.3);
    letter-spacing: 0.1em;
}
.hero-bottom {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: 18px;
}

/* ─── Danger zone ────────────────────────────────────────────────── */
.danger-zone {
    background: rgba(255,59,59,0.05);
    border: 1px solid rgba(255,59,59,0.25);
    border-left: 3px solid var(--accent-red);
    border-radius: var(--radius-card);
    padding: 14px 18px;
}
.danger-label {
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent-red);
    margin-bottom: 10px;
}

/* ─── Tag / badge ────────────────────────────────────────────────── */
.tag {
    display: inline-block;
    padding: 2px 8px;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-radius: 1px;
}
.tag-green  { background: rgba(0,255,133,0.12); color: var(--accent-neon);  border: 1px solid rgba(0,255,133,0.3); }
.tag-amber  { background: rgba(255,214,0,0.12);  color: var(--accent-amber); border: 1px solid rgba(255,214,0,0.3); }
.tag-red    { background: rgba(255,59,59,0.12);  color: var(--accent-red);   border: 1px solid rgba(255,59,59,0.3); }
.tag-ice    { background: rgba(0,223,255,0.12);  color: var(--accent-ice);   border: 1px solid rgba(0,223,255,0.3); }

/* ─── Stat row ───────────────────────────────────────────────────── */
.stat-row {
    display: flex;
    align-items: baseline;
    gap: 6px;
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-muted);
    font-size: 0.68rem;
    letter-spacing: 0.06em;
}
.stat-val {
    color: var(--accent-neon);
    font-size: 1rem;
    font-weight: 700;
}

/* ─── Tabla HTML industrial ──────────────────────────────────────── */
.ind-table-wrap {
    width: 100%;
    overflow-x: auto;
    border: 1px solid rgba(0,255,133,0.2);
    border-radius: 4px;
    margin-bottom: 0.5rem;
    animation: fadeIn 0.35s ease both;
}
.ind-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    font-family: 'Montserrat', sans-serif;
}
.ind-table thead tr {
    background: #052012;
    border-bottom: 2px solid rgba(0,255,133,0.3);
}
.ind-table thead th {
    color: #00FF85;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
    white-space: nowrap;
}
.ind-table tbody tr {
    border-bottom: 1px solid rgba(0,255,133,0.06);
    transition: background 0.15s ease;
}
.ind-table tbody tr:nth-child(even) {
    background: rgba(0,0,0,0.15);
}
.ind-table tbody tr:hover {
    background: rgba(0,255,133,0.06);
}
.ind-table tbody td {
    color: #F0FFF4;
    padding: 9px 14px;
    font-size: 0.78rem;
    font-weight: 400;
    vertical-align: middle;
    max-width: 320px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ind-empty {
    color: rgba(0,255,133,0.4);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 16px;
    text-align: center;
    border: 1px dashed rgba(0,255,133,0.15);
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# BOOTSTRAP — tabla de usuarios y auth gate
# ---------------------------------------------------------------------------
ensure_users_table()

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# ── Login screen ─────────────────────────────────────────────────────────────
if st.session_state.auth_user is None:
    st.markdown("""
    <style>
    .login-wrap {
        max-width: 400px; margin: 80px auto 0; padding: 40px 36px;
        background: var(--bg-panel); border: 1px solid var(--border-neon);
        border-radius: 6px; box-shadow: var(--shadow-neon);
    }
    .login-logo { font-size: 2.4rem; text-align: center; margin-bottom: 6px; }
    .login-title {
        font-family: 'Montserrat', sans-serif; font-weight: 800;
        font-size: 1.1rem; letter-spacing: .18em; text-transform: uppercase;
        color: var(--accent-neon); text-align: center; margin-bottom: 4px;
    }
    .login-sub {
        font-size: .72rem; color: var(--text-muted); text-align: center;
        letter-spacing: .12em; margin-bottom: 28px;
    }
    </style>
    <div class="login-wrap">
        <div class="login-logo">⬡</div>
        <div class="login-title">Poverty Stoplight</div>
        <div class="login-sub">Panel de Administración · RAG System</div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        with st.form("login_form"):
            username = st.text_input("Usuario", placeholder="usuario")
            password = st.text_input("Contraseña", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("▶ Ingresar", use_container_width=True, type="primary")

        if submitted:
            user = authenticate_user(username.strip(), password)
            if user:
                st.session_state.auth_user = user
                st.rerun()
            else:
                st.error("Credenciales incorrectas o usuario inactivo.")
    st.stop()

# ── Usuario autenticado ───────────────────────────────────────────────────────
auth_user = st.session_state.auth_user

# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
db_status = db_ok()

if db_status:
    status_html = '<span class="status-online"><span class="status-dot"></span>SISTEMA ONLINE</span>'
else:
    status_html = '<span class="status-offline"><span class="status-dot-red"></span>BD OFFLINE</span>'

col_hero, col_logout = st.columns([6, 1])
with col_hero:
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-id">SYS // RAG-ADMIN // v2.0</div>
        <div class="hero-title">⬡ Poverty Stoplight</div>
        <div class="hero-sub">Panel de Administración · Sistema RAG Multi-Embedding</div>
        <div class="hero-bottom">
            {status_html}
            <span class="tag tag-ice">PostgreSQL + pgvector</span>
            <span class="tag tag-green">Ollama Local</span>
            <span class="tag tag-amber">5 Modelos</span>
            <span class="tag tag-amber">👤 {auth_user['username']} ({auth_user['role']})</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_logout:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    if st.button("⏻ Salir", key="btn_logout"):
        st.session_state.auth_user = None
        st.rerun()

if not db_status:
    st.error("No se pudo conectar a la base de datos. Verifica que el contenedor PostgreSQL esté corriendo.")
    st.stop()

# ---------------------------------------------------------------------------
# TABS PRINCIPALES
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "▸ Dashboard",
    "▸ Documentos",
    "▸ Chunks",
    "▸ Modelos & Embeddings",
    "▸ Subir Documentos",
    "▸ Usuarios",
    "▸ Validación",
])

# ===========================================================================
# TAB 1 — DASHBOARD
# ===========================================================================
with tab1:
    n_docs    = fetchone("SELECT COUNT(*) FROM documents")[0]
    n_chunks  = fetchone("SELECT COUNT(*) FROM chunks")[0]
    n_models  = fetchone("SELECT COUNT(*) FROM embedding_models")[0]
    n_configs = fetchone("SELECT COUNT(*) FROM chunk_configs")[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documentos", n_docs)
    with col2:
        st.metric("Chunks totales", f"{n_chunks:,}")
    with col3:
        st.metric("Modelos embedding", n_models)
    with col4:
        st.metric("Configs de chunk", n_configs)

    st.divider()

    # ── Cobertura de embeddings (tarjetas personalizadas) ──────────────────
    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Cobertura de Embeddings por Modelo</div>
    </div>
    """, unsafe_allow_html=True)

    cards_html = ""
    for model_name, table_name, dims in EMBEDDING_TABLES:
        try:
            n_emb = fetchone(f"SELECT COUNT(*) FROM {table_name}")[0]
            pct   = (n_emb / n_chunks * 100) if n_chunks > 0 else 0
            cards_html += f"""
            <div class="model-card">
                <div class="model-name">{model_name}</div>
                <div class="model-dims">{dims}d</div>
                <div class="model-bar-wrap">
                    <div class="model-bar-fill" style="width:{pct:.1f}%"></div>
                </div>
                <div class="model-pct">{pct:.1f}%</div>
                <div class="model-count">{n_emb:,} / {n_chunks:,}</div>
            </div>
            """
        except Exception as e:
            cards_html += f'<div class="model-card"><div class="model-name">{model_name}</div><div style="color:var(--accent-red);font-size:0.75rem;">Error: {e}</div></div>'

    st.markdown(cards_html, unsafe_allow_html=True)

    st.divider()

    # ── Chunks por configuración ───────────────────────────────────────────
    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Chunks por Configuración</div>
    </div>
    """, unsafe_allow_html=True)

    df_cfg = query_df("""
        SELECT cc.name          AS configuracion,
               cc.chunk_size   AS tamaño_chars,
               cc.overlap      AS solapamiento,
               COUNT(c.id)     AS total_chunks
        FROM chunk_configs cc
        LEFT JOIN chunks c ON c.chunk_config_id = cc.id
        GROUP BY cc.id, cc.name, cc.chunk_size, cc.overlap
        ORDER BY cc.chunk_size
    """)
    render_table(df_cfg)

    st.divider()

    # ── Documentos recientes ───────────────────────────────────────────────
    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Documentos Recientes</div>
    </div>
    """, unsafe_allow_html=True)

    df_recent = query_df("""
        SELECT COALESCE(d.titulo, d.filename) AS titulo,
               d.filename, d.file_type AS tipo,
               COUNT(c.id) AS chunks,
               d.created_at::date AS fecha_ingesta
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.id, d.titulo, d.filename, d.file_type, d.created_at
        ORDER BY d.created_at DESC
        LIMIT 10
    """)
    render_table(df_recent)


# ===========================================================================
# TAB 2 — DOCUMENTOS
# ===========================================================================
with tab2:

    # ── session state ────────────────────────────────────────────────────────
    if "doc_edit_id" not in st.session_state:
        st.session_state.doc_edit_id = None
    if "confirm_delete_id" not in st.session_state:
        st.session_state.confirm_delete_id = None

    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("↺ Actualizar", key="refresh_docs"):
            st.session_state.doc_edit_id = None
            st.session_state.pop("doc_detail_id", None)
            st.rerun()

    df_docs = query_df("""
        SELECT d.id,
               COALESCE(d.titulo, '—')                   AS titulo,
               d.filename,
               d.file_type                               AS tipo,
               COUNT(c.id)                               AS total_chunks,
               d.created_at::date                        AS ingesta
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.id, d.titulo, d.filename, d.file_type, d.created_at
        ORDER BY d.id
    """)

    if df_docs.empty:
        st.info("No hay documentos registrados.")
    else:
        st.markdown(f"""
        <div class="ind-section">
            <div class="bar"></div>
            <div class="label">Documentos Indexados</div>
            <div class="count">{len(df_docs)} registros</div>
        </div>
        """, unsafe_allow_html=True)

        # ── tabla visual (HTML) ──────────────────────────────────────────────
        render_table(df_docs)

        st.divider()

        # ── selector de documento (radio estilizado) ─────────────────────────
        st.markdown("""
        <div class="ind-section">
            <div class="bar"></div>
            <div class="label">Seleccionar Documento</div>
        </div>
        """, unsafe_allow_html=True)

        radio_options = [
            f"#{int(r['id'])}  |  {r['titulo'] if r['titulo'] != '—' else r['filename']}"
            for _, r in df_docs.iterrows()
        ]
        id_by_option = {
            radio_options[i]: int(df_docs.iloc[i]["id"])
            for i in range(len(df_docs))
        }

        st.markdown('<div class="doc-radio">', unsafe_allow_html=True)
        selected_option = st.radio(
            "documento",
            options=radio_options,
            index=0,
            key="doc_radio_selector",
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        selected_id = id_by_option[selected_option]

        # ── botones de acción ────────────────────────────────────────────────
        col_btn_edit, col_btn_detail, col_btn_del, _spacer = st.columns([1, 1, 1, 4])
        with col_btn_edit:
            if st.button("✎ Editar", type="primary", key="btn_open_edit"):
                st.session_state.doc_edit_id       = selected_id
                st.session_state.confirm_delete_id = None
                st.session_state.pop("doc_detail_id", None)
        with col_btn_detail:
            if st.button("◉ Ver detalles", key="btn_detail"):
                st.session_state.doc_edit_id       = None
                st.session_state.confirm_delete_id = None
                df_detail = query_df("""
                    SELECT cc.name AS config, c.format,
                           COUNT(c.id) AS chunks,
                           AVG(LENGTH(c.chunk_text))::int AS avg_chars
                    FROM chunks c
                    JOIN chunk_configs cc ON c.chunk_config_id = cc.id
                    WHERE c.document_id = %s
                    GROUP BY cc.name, c.format
                    ORDER BY cc.name, c.format
                """, params=(selected_id,))
                st.session_state.doc_detail_df = df_detail
                st.session_state.doc_detail_id = selected_id
        with col_btn_del:
            if st.button("✕ Eliminar", type="secondary", key="btn_del_init"):
                st.session_state.confirm_delete_id = selected_id
                st.session_state.doc_edit_id       = None
                st.session_state.pop("doc_detail_id", None)

        # ── panel detalles ────────────────────────────────────────────────────
        if st.session_state.get("doc_detail_id"):
            detail_id = st.session_state.doc_detail_id
            st.divider()
            st.markdown(f"""
            <div class="ind-section">
                <div class="bar"></div>
                <div class="label">Detalles — Doc #{detail_id}</div>
            </div>
            """, unsafe_allow_html=True)
            render_table(st.session_state.doc_detail_df)

            n_doc_chunks = fetchone(
                "SELECT COUNT(*) FROM chunks WHERE document_id = %s", (detail_id,)
            )[0]
            emb_rows = []
            for model_name, table_name, dims in EMBEDDING_TABLES:
                try:
                    n_emb = fetchone(f"""
                        SELECT COUNT(*) FROM {table_name} e
                        JOIN chunks c ON e.chunk_id = c.id
                        WHERE c.document_id = %s
                    """, (detail_id,))[0]
                    pct = (n_emb / n_doc_chunks * 100) if n_doc_chunks > 0 else 0
                    emb_rows.append({"Modelo": model_name, "Embeddings": n_emb,
                                     "Chunks": n_doc_chunks, "Cobertura %": round(pct, 1)})
                except Exception as e:
                    emb_rows.append({"Modelo": model_name, "Error": str(e)})
            render_table(pd.DataFrame(emb_rows))

        # ── panel edición ─────────────────────────────────────────────────────
        edit_id = st.session_state.doc_edit_id
        if edit_id:
            st.divider()
            st.markdown(f"""
            <div class="ind-section">
                <div class="bar"></div>
                <div class="label">Editar Documento #{edit_id}</div>
            </div>
            """, unsafe_allow_html=True)

            doc_row = fetchone(
                "SELECT filename, titulo FROM documents WHERE id = %s", (edit_id,)
            )
            current_filename = doc_row[0] if doc_row else ""
            current_titulo   = doc_row[1] if doc_row and doc_row[1] else ""

            col_form, col_actions = st.columns([3, 1])
            with col_form:
                # Keys dinámicas por doc_id — se regeneran al cambiar documento
                new_titulo = st.text_input(
                    "Título (mostrado en el chat)",
                    value=current_titulo,
                    key=f"input_titulo_{edit_id}",
                    placeholder="Ej: Manual Metodológico v4 (2024)",
                )
                new_filename = st.text_input(
                    "Nombre de archivo",
                    value=current_filename,
                    key=f"input_rename_{edit_id}",
                )
            with col_actions:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("💾 Guardar", type="primary", key=f"btn_save_{edit_id}"):
                    t = new_titulo.strip() or None
                    f = new_filename.strip()
                    if not f:
                        st.error("El nombre de archivo no puede estar vacío.")
                    else:
                        execute_sql(
                            "UPDATE documents SET filename = %s, titulo = %s WHERE id = %s",
                            (f, t, edit_id),
                        )
                        st.success("Guardado correctamente.")
                        st.session_state.doc_edit_id = None
                        st.rerun()
                if st.button("✕ Cancelar", key=f"btn_cancel_{edit_id}"):
                    st.session_state.doc_edit_id = None
                    st.rerun()

        # ── confirmación eliminación ──────────────────────────────────────────
        del_id = st.session_state.confirm_delete_id
        if del_id:
            st.divider()
            st.warning(f"¿Eliminar doc #{del_id} y todos sus chunks y embeddings en cascada?")
            c1, c2 = st.columns([1, 5])
            with c1:
                if st.button("✓ Confirmar", type="primary", key="btn_del_confirm"):
                    execute_sql("DELETE FROM documents WHERE id = %s", (del_id,))
                    st.session_state.confirm_delete_id = None
                    st.session_state.doc_edit_id       = None
                    st.session_state.pop("doc_detail_id", None)
                    st.success(f"Doc #{del_id} eliminado.")
                    st.rerun()
            with c2:
                if st.button("✕ Cancelar eliminación", key="btn_del_cancel"):
                    st.session_state.confirm_delete_id = None
                    st.rerun()


# ===========================================================================
# TAB 3 — CHUNKS
# ===========================================================================
with tab3:
    docs_list    = query_df("SELECT id, filename FROM documents ORDER BY filename")
    configs_list = query_df("SELECT name FROM chunk_configs ORDER BY chunk_size")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        doc_opts = ["(todos)"] + [
            f"{r['id']} — {r['filename']}" for _, r in docs_list.iterrows()
        ]
        doc_filter = st.selectbox("Documento", doc_opts)

    with col_f2:
        cfg_opts    = ["(todas)"] + configs_list["name"].tolist()
        config_filter = st.selectbox("Configuración", cfg_opts)

    with col_f3:
        fmt_opts    = ["(todos)", "markdown", "plaintext"]
        format_filter = st.selectbox("Formato", fmt_opts)

    with col_f4:
        page_size = st.selectbox("Filas / página", [25, 50, 100], index=0)

    col_pag, col_stat = st.columns([1, 3])
    with col_pag:
        page_num = st.number_input("Página", min_value=1, value=1, step=1)

    conditions: list[str] = []
    params: list = []

    if doc_filter != "(todos)":
        doc_id = int(doc_filter.split(" — ")[0])
        conditions.append("c.document_id = %s")
        params.append(doc_id)

    if config_filter != "(todas)":
        conditions.append("cc.name = %s")
        params.append(config_filter)

    if format_filter != "(todos)":
        conditions.append("c.format = %s")
        params.append(format_filter)

    where  = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    offset = (page_num - 1) * page_size

    total = fetchone(
        f"SELECT COUNT(*) FROM chunks c "
        f"JOIN chunk_configs cc ON c.chunk_config_id = cc.id "
        f"JOIN documents d ON c.document_id = d.id {where}",
        tuple(params) if params else None,
    )[0]

    n_pages = max(1, -(-total // page_size))

    with col_stat:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding-top:28px;">
            <div class="stat-row"><span class="stat-val">{total:,}</span> chunks encontrados</div>
            <span class="tag tag-ice">Pág {page_num} / {n_pages}</span>
        </div>
        """, unsafe_allow_html=True)

    df_chunks = query_df(
        f"""
        SELECT c.id,
               d.filename       AS documento,
               cc.name          AS config,
               c.format,
               c.chunk_index    AS idx,
               LENGTH(c.chunk_text) AS chars,
               LEFT(c.chunk_text, 220) AS preview
        FROM chunks c
        JOIN chunk_configs cc ON c.chunk_config_id = cc.id
        JOIN documents d      ON c.document_id = d.id
        {where}
        ORDER BY d.filename, cc.name, c.format, c.chunk_index
        LIMIT %s OFFSET %s
        """,
        params=tuple(params) + (page_size, offset),
    )
    render_table(df_chunks)

    st.divider()
    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Texto Completo de Chunk</div>
    </div>
    """, unsafe_allow_html=True)

    if not df_chunks.empty:
        default_id = int(df_chunks["id"].iloc[0])
    else:
        default_id = 1

    col_id, col_btn = st.columns([2, 1])
    with col_id:
        chunk_id_input = st.number_input("ID del chunk", min_value=1, value=default_id)
    with col_btn:
        st.markdown('<div style="padding-top:24px;">', unsafe_allow_html=True)
        show = st.button("▶ Mostrar chunk", key="btn_show_chunk")
        st.markdown('</div>', unsafe_allow_html=True)

    if show:
        row = fetchone("SELECT chunk_text, document_id, format FROM chunks WHERE id = %s", (chunk_id_input,))
        if row:
            st.text_area("Texto del chunk", value=row[0], height=350)
            st.markdown(f"""
            <div style="display:flex;gap:10px;margin-top:4px;">
                <span class="tag tag-green">document_id: {row[1]}</span>
                <span class="tag tag-ice">format: {row[2]}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"No se encontró el chunk #{chunk_id_input}.")


# ===========================================================================
# TAB 4 — MODELOS & EMBEDDINGS
# ===========================================================================
with tab4:
    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Modelos Registrados en BD</div>
    </div>
    """, unsafe_allow_html=True)

    df_models = query_df("""
        SELECT model_name  AS modelo,
               table_name  AS tabla,
               dimensions  AS dimensiones,
               created_at::date AS registrado
        FROM embedding_models
        ORDER BY id
    """)
    render_table(df_models)

    st.divider()

    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Cobertura Global de Embeddings</div>
    </div>
    """, unsafe_allow_html=True)

    n_total = fetchone("SELECT COUNT(*) FROM chunks")[0]
    cov_rows = []
    cards2_html = ""
    for model_name, table_name, dims in EMBEDDING_TABLES:
        try:
            n_emb  = fetchone(f"SELECT COUNT(*) FROM {table_name}")[0]
            skips  = n_total - n_emb
            pct    = (n_emb / n_total * 100) if n_total > 0 else 0.0
            cov_rows.append({
                "Modelo":         model_name,
                "Tabla":          table_name,
                "Dimensiones":    dims,
                "Embeddings":     n_emb,
                "Chunks totales": n_total,
                "Omitidos":       skips,
                "Cobertura %":    round(pct, 2),
            })
            skip_tag = f'<span class="tag tag-amber">{skips} omitidos</span>' if skips > 0 else '<span class="tag tag-green">0 omitidos</span>'
            cards2_html += f"""
            <div class="model-card">
                <div class="model-name">{model_name}</div>
                <div class="model-dims">{dims}d</div>
                <div class="model-bar-wrap">
                    <div class="model-bar-fill" style="width:{pct:.1f}%"></div>
                </div>
                <div class="model-pct">{pct:.1f}%</div>
                <div class="model-count">{n_emb:,} / {n_total:,}</div>
                {skip_tag}
            </div>
            """
        except Exception as e:
            cov_rows.append({"Modelo": model_name, "Error": str(e)})

    st.markdown(cards2_html, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Configuraciones de Chunk</div>
    </div>
    """, unsafe_allow_html=True)

    df_cfgs = query_df("""
        SELECT cc.name         AS nombre,
               cc.chunk_size  AS tamaño_chars,
               cc.overlap     AS solapamiento,
               (SELECT COUNT(*) FROM chunks c WHERE c.chunk_config_id = cc.id) AS chunks_generados
        FROM chunk_configs cc
        ORDER BY cc.chunk_size
    """)
    render_table(df_cfgs)

    st.divider()

    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Embeddings por Documento (bge-m3)</div>
    </div>
    """, unsafe_allow_html=True)

    df_cross = query_df("""
        SELECT d.filename,
               d.file_type                               AS tipo,
               COUNT(c.id)                               AS chunks_totales,
               COUNT(e.id)                               AS embeddings_bge_m3
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        LEFT JOIN embeddings_bge_m3 e ON e.chunk_id = c.id
        GROUP BY d.id, d.filename, d.file_type
        ORDER BY d.filename
    """)
    render_table(df_cross)


# ===========================================================================
# TAB 5 — SUBIR DOCUMENTOS
# ===========================================================================
with tab5:
    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Subir Nuevos Documentos al Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:10px;margin-bottom:1.2rem;">
        <span class="tag tag-green">PDF</span>
        <span class="tag tag-ice">CSV</span>
        <span class="tag tag-amber">Markdown</span>
        <span style="font-size:0.72rem;color:var(--text-muted,#4ADE80);margin-left:4px;align-self:center;">
            Docling + 5 modelos de embedding. Puede tardar varios minutos.
        </span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Seleccionar archivos",
        type=["pdf", "csv", "md"],
        accept_multiple_files=True,
        key="uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"""
        <div class="ind-section">
            <div class="bar"></div>
            <div class="label">Archivos seleccionados</div>
            <div class="count">{len(uploaded_files)} archivo(s)</div>
        </div>
        """, unsafe_allow_html=True)

        for uf in uploaded_files:
            already_in_data = (DATA_DIR / uf.name).exists()
            already_in_db   = bool(fetchone("SELECT 1 FROM documents WHERE filename = %s", (uf.name,)))
            flags_html = ""
            if already_in_data:
                flags_html += '<span class="tag tag-amber" style="margin-left:8px;">ya en /data</span>'
            if already_in_db:
                flags_html += '<span class="tag tag-red" style="margin-left:4px;">ya en BD</span>'
            size_kb = f"{uf.size / 1024:.1f} KB"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:6px;padding:6px 0;border-bottom:1px solid rgba(0,255,133,0.06);">
                <span style="color:var(--accent-neon,#00FF85);font-size:0.8rem;">▸</span>
                <span style="font-size:0.8rem;font-weight:600;color:var(--text-primary,#F0FFF4);">{uf.name}</span>
                <span class="tag tag-ice">{size_kb}</span>
                {flags_html}
            </div>
            """, unsafe_allow_html=True)

        run_ingest = st.checkbox("Ejecutar ingesta automáticamente tras guardar", value=True)

        if st.button("▶ Guardar y procesar", type="primary", key="btn_upload"):
            saved = []
            for uf in uploaded_files:
                dest = DATA_DIR / uf.name
                try:
                    dest.write_bytes(uf.getvalue())
                    saved.append(uf.name)
                    st.success(f"Guardado: `{uf.name}`")
                except Exception as e:
                    st.error(f"Error guardando `{uf.name}`: {e}")

            if saved and run_ingest:
                st.divider()
                st.markdown("""
                <div class="ind-section">
                    <div class="bar"></div>
                    <div class="label">Pipeline de Ingesta</div>
                </div>
                """, unsafe_allow_html=True)

                ingest_script = PROJECT_ROOT / "scripts" / "ingest_all.py"
                env = os.environ.copy()

                with st.spinner("Ejecutando ingest_all.py …"):
                    try:
                        result = subprocess.run(
                            [sys.executable, str(ingest_script)],
                            capture_output=True,
                            text=True,
                            cwd=str(PROJECT_ROOT),
                            env=env,
                            timeout=1800,
                        )

                        log_text = result.stdout
                        if result.stderr:
                            log_text += "\n\n--- STDERR ---\n" + result.stderr

                        st.text_area("Log de ingesta", value=log_text, height=450)

                        if result.returncode == 0:
                            st.success("Ingesta completada exitosamente.")
                            st.rerun()
                        else:
                            st.error(f"Proceso terminó con código {result.returncode}.")

                    except subprocess.TimeoutExpired:
                        st.error("Tiempo límite (30 min) excedido. Ejecuta la ingesta manualmente.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.divider()

    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Estado de Archivos en /data</div>
    </div>
    """, unsafe_allow_html=True)

    supported_exts = {".pdf", ".csv", ".md"}
    file_rows = []
    for f in sorted(DATA_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in supported_exts:
            in_db  = bool(fetchone("SELECT 1 FROM documents WHERE filename = %s", (f.name,)))
            n_chunks_f = fetchone(
                "SELECT COUNT(*) FROM chunks c JOIN documents d ON c.document_id = d.id WHERE d.filename = %s",
                (f.name,)
            )
            file_rows.append({
                "Archivo":     f.name,
                "Tipo":        f.suffix.lstrip(".").upper(),
                "Tamaño (KB)": round(f.stat().st_size / 1024, 1),
                "En BD":       "✅" if in_db else "❌",
                "Chunks":      n_chunks_f[0] if n_chunks_f else 0,
            })

    if file_rows:
        render_table(pd.DataFrame(file_rows))
    else:
        st.info("No se encontraron archivos soportados en `/data`.")


# ===========================================================================
# TAB 6 — USUARIOS
# ===========================================================================
with tab6:
    if auth_user["role"] != "admin":
        st.warning("Solo los administradores pueden gestionar usuarios.")
        st.stop()

    if "usr_edit_id" not in st.session_state:
        st.session_state.usr_edit_id = None
    if "usr_confirm_del" not in st.session_state:
        st.session_state.usr_confirm_del = None

    col_ref, _ = st.columns([1, 5])
    with col_ref:
        if st.button("↺ Actualizar", key="refresh_users"):
            st.session_state.usr_edit_id    = None
            st.session_state.usr_confirm_del = None
            st.rerun()

    # ── tabla de usuarios ─────────────────────────────────────────────────
    df_users = query_df("""
        SELECT id,
               username,
               COALESCE(email, '—') AS email,
               role,
               CASE WHEN is_active THEN '✅' ELSE '❌' END AS activo,
               created_at::date AS creado
        FROM admin_users
        ORDER BY id
    """)

    st.markdown(f"""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Usuarios Registrados</div>
        <div class="count">{len(df_users)} registros</div>
    </div>
    """, unsafe_allow_html=True)
    render_table(df_users)

    st.divider()

    # ── selector + botones ────────────────────────────────────────────────
    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Seleccionar Usuario</div>
    </div>
    """, unsafe_allow_html=True)

    usr_options = [
        f"#{int(r['id'])}  |  {r['username']}  [{r['role']}]"
        for _, r in df_users.iterrows()
    ]
    uid_by_option = {
        usr_options[i]: int(df_users.iloc[i]["id"])
        for i in range(len(df_users))
    }

    st.markdown('<div class="doc-radio">', unsafe_allow_html=True)
    selected_usr_opt = st.radio(
        "usuario",
        options=usr_options,
        index=0,
        key="usr_radio_selector",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    sel_uid = uid_by_option[selected_usr_opt]

    col_ue, col_ud, _sp = st.columns([1, 1, 5])
    with col_ue:
        if st.button("✎ Editar", type="primary", key="btn_usr_edit"):
            st.session_state.usr_edit_id    = sel_uid
            st.session_state.usr_confirm_del = None
    with col_ud:
        if st.button("✕ Eliminar", type="secondary", key="btn_usr_del"):
            if sel_uid == auth_user["id"]:
                st.error("No puedes eliminar tu propia cuenta.")
            else:
                st.session_state.usr_confirm_del = sel_uid
                st.session_state.usr_edit_id    = None

    # ── panel edición usuario ─────────────────────────────────────────────
    uedit_id = st.session_state.usr_edit_id
    if uedit_id:
        st.divider()
        st.markdown(f"""
        <div class="ind-section">
            <div class="bar"></div>
            <div class="label">Editar Usuario #{uedit_id}</div>
        </div>
        """, unsafe_allow_html=True)

        urow = fetchone(
            "SELECT username, email, role, is_active FROM admin_users WHERE id = %s",
            (uedit_id,),
        )
        cur_username, cur_email, cur_role, cur_active = urow if urow else ("", "", "viewer", True)

        col_ef, col_ea = st.columns([3, 1])
        with col_ef:
            new_uname  = st.text_input("Usuario",  value=cur_username or "",  key=f"ue_name_{uedit_id}")
            new_email  = st.text_input("Email",    value=cur_email or "",     key=f"ue_email_{uedit_id}")
            new_role   = st.selectbox("Rol", ["admin", "viewer"],
                                      index=0 if cur_role == "admin" else 1,
                                      key=f"ue_role_{uedit_id}")
            new_active = st.checkbox("Activo", value=bool(cur_active), key=f"ue_active_{uedit_id}")
            st.markdown("**Cambiar contraseña** *(dejar vacío para no cambiar)*")
            new_pw1 = st.text_input("Nueva contraseña",   type="password", key=f"ue_pw1_{uedit_id}")
            new_pw2 = st.text_input("Repetir contraseña", type="password", key=f"ue_pw2_{uedit_id}")

        with col_ea:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            if st.button("💾 Guardar", type="primary", key=f"ue_save_{uedit_id}"):
                errors = []
                if not new_uname.strip():
                    errors.append("El nombre de usuario no puede estar vacío.")
                if new_pw1 and new_pw1 != new_pw2:
                    errors.append("Las contraseñas no coinciden.")
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    if new_pw1:
                        ph, salt = hash_password(new_pw1)
                        execute_sql(
                            "UPDATE admin_users SET username=%s, email=%s, role=%s, "
                            "is_active=%s, password_hash=%s, salt=%s WHERE id=%s",
                            (new_uname.strip(), new_email.strip() or None,
                             new_role, new_active, ph, salt, uedit_id),
                        )
                    else:
                        execute_sql(
                            "UPDATE admin_users SET username=%s, email=%s, role=%s, is_active=%s WHERE id=%s",
                            (new_uname.strip(), new_email.strip() or None, new_role, new_active, uedit_id),
                        )
                    st.success("Usuario actualizado correctamente.")
                    st.session_state.usr_edit_id = None
                    st.rerun()
            if st.button("✕ Cancelar", key=f"ue_cancel_{uedit_id}"):
                st.session_state.usr_edit_id = None
                st.rerun()

    # ── confirmación eliminación ──────────────────────────────────────────
    udel_id = st.session_state.usr_confirm_del
    if udel_id:
        st.divider()
        st.warning(f"¿Eliminar usuario #{udel_id}? Esta acción no se puede deshacer.")
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("✓ Confirmar", type="primary", key="btn_udel_confirm"):
                execute_sql("DELETE FROM admin_users WHERE id = %s", (udel_id,))
                st.session_state.usr_confirm_del = None
                st.success(f"Usuario #{udel_id} eliminado.")
                st.rerun()
        with c2:
            if st.button("✕ Cancelar", key="btn_udel_cancel"):
                st.session_state.usr_confirm_del = None
                st.rerun()

    st.divider()

    # ── crear nuevo usuario ─────────────────────────────────────────────
    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Crear Nuevo Usuario</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_new_user"):
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            nu_username = st.text_input("Usuario *", placeholder="nombre_usuario")
            nu_password = st.text_input("Contraseña *", type="password")
        with col_n2:
            nu_email    = st.text_input("Email", placeholder="user@ejemplo.com")
            nu_role     = st.selectbox("Rol", ["viewer", "admin"])
        nu_submit = st.form_submit_button("▶ Crear Usuario", type="primary")

    if nu_submit:
        errors = []
        if not nu_username.strip():
            errors.append("El nombre de usuario es obligatorio.")
        if not nu_password:
            errors.append("La contraseña es obligatoria.")
        if fetchone("SELECT 1 FROM admin_users WHERE username = %s", (nu_username.strip(),)):
            errors.append(f"El usuario '{nu_username.strip()}' ya existe.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            ph, salt = hash_password(nu_password)
            execute_sql(
                "INSERT INTO admin_users (username, email, password_hash, salt, role) VALUES (%s,%s,%s,%s,%s)",
                (nu_username.strip(), nu_email.strip() or None, ph, salt, nu_role),
            )
            st.success(f"Usuario '{nu_username.strip()}' creado con rol '{nu_role}'.")
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — Validación
# ═══════════════════════════════════════════════════════════════════════════

with tab7:
    if "val_sel_run" not in st.session_state:
        st.session_state.val_sel_run = None
    if "val_category" not in st.session_state:
        st.session_state.val_category = "Todas"
    if "val_page" not in st.session_state:
        st.session_state.val_page = 0
    if "val_version_id" not in st.session_state:
        st.session_state.val_version_id = None

    # ── selector de versión ───────────────────────────────────────────────
    df_versions = query_df(
        """SELECT va.id, va.numero, va.descripcion, va.activa,
                  lm.model_name AS llm, em.model_name AS embed, cc.name AS chunk
           FROM version_agente va
           JOIN llm_models lm       ON lm.id = va.llm_model_id
           JOIN embedding_models em ON em.id = va.embed_model_id
           JOIN chunk_configs cc    ON cc.id = va.chunk_config_id
           ORDER BY va.numero"""
    )

    if df_versions.empty:
        st.error("No hay versiones de agente registradas. Ejecuta la migración 005.")
        st.stop()

    # Opciones del selector: "v1 — Versión inicial (activa)"
    def ver_label(r):
        tag = " ✦ activa" if r["activa"] else ""
        return f"v{int(r['numero'])} — {r['descripcion']}{tag}"

    ver_options = [ver_label(r) for _, r in df_versions.iterrows()]
    ver_id_map  = {ver_options[i]: int(df_versions.iloc[i]["id"]) for i in range(len(df_versions))}

    # Default: versión activa
    default_idx = next(
        (i for i, r in df_versions.iterrows() if r["activa"]), 0
    )
    if st.session_state.val_version_id is None:
        st.session_state.val_version_id = int(df_versions.iloc[default_idx]["id"])

    prev_ver_id = st.session_state.val_version_id
    cur_ver_opt = next(
        (o for o in ver_options if ver_id_map[o] == st.session_state.val_version_id),
        ver_options[default_idx],
    )

    col_ver, col_ver_info = st.columns([2, 3])
    with col_ver:
        sel_ver_opt = st.selectbox(
            "Versión del agente a validar",
            options=ver_options,
            index=ver_options.index(cur_ver_opt),
            key="val_version_select",
        )
    st.session_state.val_version_id = ver_id_map[sel_ver_opt]
    if st.session_state.val_version_id != prev_ver_id:
        st.session_state.val_page = 0
        st.rerun()

    va_row = df_versions[df_versions["id"] == st.session_state.val_version_id].iloc[0]
    va_id    = int(va_row["id"])
    va_num   = int(va_row["numero"])
    va_desc  = va_row["descripcion"]
    va_llm   = va_row["llm"]
    va_embed = va_row["embed"]
    va_chunk = va_row["chunk"]
    va_activa = bool(va_row["activa"])

    with col_ver_info:
        badge_color = "#00FF85" if va_activa else "#f59e0b"
        badge_text  = "VERSIÓN POR DEFECTO" if va_activa else "VERSIÓN ANTERIOR"
        st.markdown(
            f"<div style='margin-top:28px;padding:8px 14px;background:#111;border:1px solid {badge_color}40;"
            f"border-left:3px solid {badge_color};border-radius:6px;font-size:.82rem'>"
            f"<span style='color:{badge_color};font-weight:700'>{badge_text}</span>"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;LLM: <code>{va_llm}</code>"
            f"&nbsp;·&nbsp;Embed: <code>{va_embed}</code>"
            f"&nbsp;·&nbsp;Chunk: <code>{va_chunk}</code>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── filtro categoría ──────────────────────────────────────────────────
    cats_raw = query_df("SELECT DISTINCT category FROM knowledge_base WHERE category IS NOT NULL ORDER BY category")
    cat_options = ["Todas"] + cats_raw["category"].tolist()
    prev_cat = st.session_state.val_category
    st.session_state.val_category = st.selectbox(
        "Filtrar por categoría",
        cat_options,
        index=cat_options.index(st.session_state.val_category)
        if st.session_state.val_category in cat_options else 0,
        key="val_cat_select",
    )
    if st.session_state.val_category != prev_cat:
        st.session_state.val_page = 0
    cat_filter = st.session_state.val_category

    # ── cargar preguntas con estadísticas ─────────────────────────────────
    cat_clause = "AND kb.category = %(cat)s" if cat_filter != "Todas" else ""
    df_q = query_df(f"""
        SELECT
            kb.id                                          AS kb_id,
            kb.question,
            COALESCE(kb.category, '—')                    AS categoria,
            er.id                                         AS eval_run_id,
            COUNT(v.id)                                   AS evaluaciones,
            ROUND(AVG(v.calificacion)::numeric, 1)        AS prom,
            SUM(CASE WHEN v.alucinacion THEN 1 ELSE 0 END) AS alucinaciones,
            MAX(CASE WHEN v.user_id = %(uid)s THEN v.calificacion END) AS mi_nota
        FROM knowledge_base kb
        JOIN eval_runs er
          ON er.knowledge_base_id = kb.id
         AND er.llm_model_id    = (SELECT llm_model_id    FROM version_agente WHERE id = %(va_id)s)
         AND er.embedding_model_id = (SELECT embed_model_id FROM version_agente WHERE id = %(va_id)s)
         AND er.chunk_config_id = (SELECT chunk_config_id FROM version_agente WHERE id = %(va_id)s)
        LEFT JOIN validaciones v ON v.eval_run_id = er.id
        WHERE 1=1 {cat_clause}
        GROUP BY kb.id, kb.question, kb.category, er.id
        ORDER BY kb.id
    """, params={"uid": auth_user["id"], "va_id": va_id, "cat": cat_filter if cat_filter != "Todas" else None})

    PAGE_SIZE = 20
    total_q   = len(df_q)
    total_pages = max(1, (total_q + PAGE_SIZE - 1) // PAGE_SIZE)
    st.session_state.val_page = min(st.session_state.val_page, total_pages - 1)
    page = st.session_state.val_page
    df_page = df_q.iloc[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

    st.markdown(f"""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Preguntas</div>
        <div class="count">{total_q} preguntas · {int(df_q['evaluaciones'].sum())} evaluaciones</div>
    </div>
    """, unsafe_allow_html=True)

    # ── tabla resumen (página actual) ────────────────────────────────────
    def render_val_table(df):
        rows = ""
        for _, r in df.iterrows():
            nota  = f"<span style='color:#00FF85;font-weight:700'>{int(r['mi_nota'])}/10</span>" if pd.notna(r["mi_nota"]) else "<span style='opacity:.4'>—</span>"
            aluci = f"<span style='color:#ff4b4b;font-weight:600'>{int(r['alucinaciones'])}</span>" if r["alucinaciones"] > 0 else "<span style='opacity:.4'>0</span>"
            prom  = f"{r['prom']}" if pd.notna(r["prom"]) else "—"
            pregunta = str(r["question"])[:80] + ("…" if len(str(r["question"])) > 80 else "")
            rows += (
                f"<tr>"
                f"<td style='color:#aaa'>{int(r['kb_id'])}</td>"
                f"<td style='max-width:380px'>{pregunta}</td>"
                f"<td style='text-align:center'>{r['categoria']}</td>"
                f"<td style='text-align:center'>{int(r['evaluaciones'])}</td>"
                f"<td style='text-align:center'>{prom}</td>"
                f"<td style='text-align:center'>{aluci}</td>"
                f"<td style='text-align:center'>{nota}</td>"
                f"</tr>"
            )
        st.markdown(f"""
        <div style='overflow-x:auto'>
        <table style='width:100%;border-collapse:collapse;font-size:.82rem;font-family:monospace'>
          <thead>
            <tr style='border-bottom:1px solid #333;color:#888;text-transform:uppercase;font-size:.72rem'>
              <th style='text-align:left;padding:6px 8px'>#</th>
              <th style='text-align:left;padding:6px 8px'>Pregunta</th>
              <th style='text-align:center;padding:6px 8px'>Categoría</th>
              <th style='text-align:center;padding:6px 8px'>Evaluaciones</th>
              <th style='text-align:center;padding:6px 8px'>Promedio</th>
              <th style='text-align:center;padding:6px 8px'>Alucinaciones</th>
              <th style='text-align:center;padding:6px 8px'>Mi nota</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    render_val_table(df_page)

    # ── paginación ────────────────────────────────────────────────────────
    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀ Anterior", disabled=page == 0, key="val_prev"):
            st.session_state.val_page -= 1
            st.rerun()
    with col_info:
        st.markdown(
            f"<div style='text-align:center;font-size:.82rem;padding-top:6px'>"
            f"Página {page + 1} de {total_pages} · {total_q} preguntas</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Siguiente ▶", disabled=page >= total_pages - 1, key="val_next"):
            st.session_state.val_page += 1
            st.rerun()

    st.divider()

    # ── selector de pregunta (combobox) ───────────────────────────────────
    st.markdown("""
    <div class="ind-section">
        <div class="bar"></div>
        <div class="label">Seleccionar Pregunta para Evaluar</div>
    </div>
    """, unsafe_allow_html=True)

    q_options = [
        f"#{int(r['kb_id'])}  |  {str(r['question'])[:90]}"
        for _, r in df_q.iterrows()
    ]
    run_by_opt = {q_options[i]: int(df_q.iloc[i]["eval_run_id"]) for i in range(len(df_q))}
    kb_by_opt  = {q_options[i]: int(df_q.iloc[i]["kb_id"])      for i in range(len(df_q))}

    sel_q_opt = st.selectbox(
        "Pregunta",
        options=q_options,
        index=0,
        key="val_q_select",
        label_visibility="collapsed",
    )

    sel_run_id = run_by_opt[sel_q_opt]
    sel_kb_id  = kb_by_opt[sel_q_opt]

    # ── panel de evaluación ───────────────────────────────────────────────
    st.divider()

    badge_color2 = "#00FF85" if va_activa else "#f59e0b"
    badge_text2  = "versión por defecto" if va_activa else "versión anterior"
    st.markdown(
        f"<div style='margin-bottom:12px;font-size:.80rem;color:#888'>"
        f"Evaluando respuestas de "
        f"<span style='color:{badge_color2};font-weight:700'>v{va_num} — {va_desc}</span>"
        f"&nbsp;<span style='background:{badge_color2}22;color:{badge_color2};"
        f"border:1px solid {badge_color2}55;border-radius:4px;padding:1px 7px;"
        f"font-size:.72rem;font-weight:600'>{badge_text2}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    kb_row = fetchone(
        "SELECT question, answer FROM knowledge_base WHERE id = %s", (sel_kb_id,)
    )
    er_row = fetchone(
        "SELECT generated_answer FROM eval_runs WHERE id = %s", (sel_run_id,)
    )

    if kb_row and er_row:
        question_text, ground_truth = kb_row
        agent_answer = er_row[0] or "—"

        # ── pregunta ──────────────────────────────────────────────────────
        st.markdown("**Pregunta**")
        st.info(question_text)

        # ── fila 1: respuesta esperada | respuesta del agente ─────────────
        col_gt, col_ag = st.columns(2)

        with col_gt:
            st.markdown("**Respuesta esperada (Ground Truth)**")
            st.markdown(
                f"<div style='background:#111;border:1px solid #333;border-radius:6px;"
                f"padding:10px 14px;font-size:.83rem;line-height:1.6;min-height:180px'>"
                f"{ground_truth}</div>",
                unsafe_allow_html=True,
            )

        with col_ag:
            st.markdown("**Respuesta del Agente**")
            st.markdown(
                f"<div style='background:#0d1f17;border:1px solid #00FF8540;"
                f"border-radius:6px;padding:10px 14px;font-size:.83rem;line-height:1.6;min-height:180px'>"
                f"{agent_answer}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

        # ── fila 2: métricas RAGAS | formulario de evaluación ────────────
        col_ragas, col_form = st.columns(2)

        # RAGAS helper (defined once, used in col_ragas)
        def ragas_label(value, metric):
            if value is None:
                return "—", "#555", "Sin datos"
            v = float(value)
            if metric in ("faithfulness", "answer_relevancy"):
                if v >= 0.80:   return f"{v:.2f}", "#00FF85", "Excelente"
                elif v >= 0.60: return f"{v:.2f}", "#a3e635", "Bueno"
                elif v >= 0.40: return f"{v:.2f}", "#f59e0b", "Moderado"
                else:           return f"{v:.2f}", "#ff4b4b", "Deficiente"
            elif metric == "context_precision":
                if v >= 0.80:   return f"{v:.2f}", "#00FF85", "Excelente"
                elif v >= 0.50: return f"{v:.2f}", "#a3e635", "Bueno"
                elif v >= 0.30: return f"{v:.2f}", "#f59e0b", "Moderado"
                else:           return f"{v:.2f}", "#ff4b4b", "Deficiente"
            else:  # context_recall
                if v >= 0.80:   return f"{v:.2f}", "#00FF85", "Excelente"
                elif v >= 0.50: return f"{v:.2f}", "#a3e635", "Bueno"
                elif v >= 0.30: return f"{v:.2f}", "#f59e0b", "Aceptable"
                else:           return f"{v:.2f}", "#ff4b4b", "Deficiente"

        with col_ragas:
            st.markdown("**Métricas RAGAS**")
            ragas_row = fetchone(
                """SELECT faithfulness, answer_relevancy, context_precision, context_recall
                   FROM eval_scores WHERE eval_run_id = %s AND status = 'success'""",
                (sel_run_id,),
            )
            if ragas_row:
                faith_v, faith_c, faith_l = ragas_label(ragas_row[0], "faithfulness")
                rel_v,   rel_c,   rel_l   = ragas_label(ragas_row[1], "answer_relevancy")
                prec_v,  prec_c,  prec_l  = ragas_label(ragas_row[2], "context_precision")
                rec_v,   rec_c,   rec_l   = ragas_label(ragas_row[3], "context_recall")
                vals = [float(x) for x in ragas_row if x is not None]
                avg_num = sum(vals) / len(vals) if vals else None
                avg_v, avg_c, avg_l = ragas_label(avg_num, "faithfulness")

                def metric_card(label, val, color, nivel, subtitle=""):
                    return (
                        f"<div style='background:#111;border:1px solid #222;border-radius:8px;"
                        f"padding:10px 12px;text-align:center'>"
                        f"<div style='font-size:.68rem;color:#888;text-transform:uppercase;margin-bottom:3px'>{label}</div>"
                        f"<div style='font-size:1.4rem;font-weight:700;color:{color}'>{val}</div>"
                        f"<div style='font-size:.72rem;color:{color};margin-top:2px'>{nivel}</div>"
                        f"<div style='font-size:.65rem;color:#555;margin-top:2px'>{subtitle}</div>"
                        f"</div>"
                    )

                cards_row1 = (
                    metric_card("Avg Score",      avg_v,   avg_c,   avg_l,   "") +
                    metric_card("Faithfulness",   faith_v, faith_c, faith_l, "Sin alucinaciones") +
                    metric_card("Ans. Relevancy", rel_v,   rel_c,   rel_l,   "Relevancia respuesta")
                )
                cards_row2 = (
                    metric_card("Ctx Precision",  prec_v,  prec_c,  prec_l,  "Ranking contexto") +
                    metric_card("Ctx Recall",     rec_v,   rec_c,   rec_l,   "Cobertura contexto")
                )
                st.markdown(
                    f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:8px'>{cards_row1}</div>"
                    f"<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:8px'>{cards_row2}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Sin métricas RAGAS para esta pregunta.")

        with col_form:
            st.markdown("**Tu evaluación**")

            existing = fetchone(
                "SELECT id, calificacion, observacion, alucinacion FROM validaciones "
                "WHERE eval_run_id = %s AND user_id = %s",
                (sel_run_id, auth_user["id"]),
            )
            ex_id   = existing[0] if existing else None
            ex_nota = int(existing[1]) if existing else 5
            ex_obs  = existing[2] or "" if existing else ""
            ex_aluc = bool(existing[3]) if existing else False

            with st.form(key=f"form_val_{sel_run_id}"):
                calificacion = st.slider(
                    "Calificación (1 = muy mala · 10 = excelente)",
                    min_value=1, max_value=10, value=ex_nota,
                    key=f"slider_cal_{sel_run_id}",
                )
                alucinacion = st.toggle(
                    "¿Alucinación?",
                    value=ex_aluc,
                    key=f"toggle_aluc_{sel_run_id}",
                )
                if alucinacion:
                    st.markdown(
                        "<span style='color:#ff4b4b;font-size:.82rem'>⚠️ <b>Sí</b> — la respuesta contiene información inventada o incorrecta</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='color:#00FF85;font-size:.82rem'>✅ <b>No</b> — la respuesta está basada en el contexto</span>",
                        unsafe_allow_html=True,
                    )
                observacion = st.text_area(
                    "Observación",
                    value=ex_obs,
                    height=120,
                    placeholder="Comentarios sobre la respuesta del agente...",
                    key=f"obs_{sel_run_id}",
                )
                submitted = st.form_submit_button(
                    "💾 Guardar evaluación", type="primary", use_container_width=True
                )

            if submitted:
                if ex_id:
                    execute_sql(
                        "UPDATE validaciones SET calificacion=%s, observacion=%s, "
                        "alucinacion=%s, updated_at=NOW() WHERE id=%s",
                        (calificacion, observacion.strip() or None, alucinacion, ex_id),
                    )
                    st.success("Evaluación actualizada.")
                else:
                    execute_sql(
                        "INSERT INTO validaciones "
                        "(version_agente_id, eval_run_id, user_id, calificacion, observacion, alucinacion) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (va_id, sel_run_id, auth_user["id"],
                         calificacion, observacion.strip() or None, alucinacion),
                    )
                    st.success("Evaluación registrada.")
                st.rerun()

            if existing:
                st.caption(f"Ya evaluaste esta respuesta: **{ex_nota}/10**")

        # ── evaluaciones de otros usuarios ────────────────────────────────
        df_other = query_df("""
            SELECT au.username, v.calificacion, v.alucinacion,
                   COALESCE(v.observacion, '—') AS observacion,
                   v.updated_at::date AS fecha
            FROM validaciones v
            JOIN admin_users au ON au.id = v.user_id
            WHERE v.eval_run_id = %s
            ORDER BY v.updated_at DESC
        """, params=(sel_run_id,))

        if not df_other.empty:
            st.divider()
            st.markdown(f"**Evaluaciones registradas ({len(df_other)})**")
            df_other["alucinacion"] = df_other["alucinacion"].apply(
                lambda x: "⚠️ Sí" if x else "✅ No"
            )
            render_table(df_other)
