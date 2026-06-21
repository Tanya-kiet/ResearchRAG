"""
app.py — ResearchRAG Streamlit application.

Layout
------
Sidebar  : LLM status, indexed file list, evaluation summary
Main     : Upload → Process → Query → Results (comparison dashboard)
"""

from __future__ import annotations

import os
import sys

# ── ensure project root is importable ───────────────────────────────────────
# Streamlit must be launched from this file's directory for the bare
# `from rag.pipeline import ...`-style imports below to resolve. If app.py is
# instead invoked via a relative/absolute path from a different working
# directory (e.g. `streamlit run path/to/app.py` from elsewhere), this
# directory may not be on sys.path, causing `ModuleNotFoundError: No module
# named 'rag.pipeline'` (or 'evaluation', 'utils', etc). This guard adds the
# project root explicitly so the app works regardless of cwd. It is a no-op
# when the root is already on sys.path (the normal case).
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from config import cfg
from evaluation.logger import EvaluationLogger
from evaluation.metrics import EvaluationScore, METRIC_NAMES, METRIC_DESCRIPTIONS
from rag.pipeline import RAGPipeline, QueryResult
from utils.helpers import get_llm_from_config, save_uploaded_files, format_score_badge
from utils.text_cleaning import truncate_text

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchRAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ─── Tokens ────────────────────────────────────────────────────────────────── */
:root {
    --bg:          #FFFFFF;
    --bg-2:        #F8FAFC;
    --surface:     #FFFFFF;
    --border:      #E2E8F0;
    --border-2:    #CBD5E1;
    --accent:      #2563EB;
    --accent-dark: #1D4ED8;
    --accent-tint: #EFF6FF;
    --accent-lt:   rgba(37,99,235,0.08);
    --accent-md:   rgba(37,99,235,0.18);
    --accent-glow: rgba(37,99,235,0.39);
    --text-1:      #0F172A;
    --text-2:      #64748B;
    --text-3:      #94A3B8;
    --shadow-sm: 0 1px 2px rgba(15,23,42,0.04), 0 1px 3px rgba(15,23,42,0.06);
    --shadow-md: 0 1px 3px rgba(15,23,42,0.05), 0 10px 24px rgba(15,23,42,0.07);
    --radius:    14px;
    --sans:      'Inter', system-ui, -apple-system, sans-serif;
}

/* ─── Base ──────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: var(--sans); }
.stApp { background-color: var(--bg); color: var(--text-1); }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2.8rem 6rem 2.8rem !important;
    max-width: 1280px;
}

/* ─── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: var(--text-2); font-size: 0.8rem; line-height: 1.6;
}
[data-testid="stSidebar"] h2 {
    color: var(--text-1); font-family: var(--sans);
    font-size: 1rem; font-weight: 800; letter-spacing: -0.01em;
}
[data-testid="stSidebar"] h3 {
    color: var(--text-3); font-family: var(--sans);
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-top: 1.4rem; margin-bottom: 0.6rem;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border); margin: 0.8rem 0;
}

/* Sidebar status row */
.sb-status {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 0.78rem; color: var(--text-2);
    padding: 0.2rem 0; line-height: 1.4;
}
.sb-dot {
    width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
}
.sb-dot-green  { background: #10B981; }
.sb-dot-amber  { background: #F59E0B; }

/* Sidebar file rows */
.sb-file {
    display: flex; align-items: center; gap: 0.45rem;
    padding: 0.3rem 0.5rem; border-radius: 7px;
    color: var(--text-2); font-size: 0.77rem;
    transition: background 0.15s, color 0.15s; cursor: default;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.sb-file:hover { background: var(--bg-2); color: var(--text-1); }
.sb-file svg { flex-shrink: 0; color: var(--accent); }

/* Sidebar eval */
.sb-eval-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.25rem 0; border-bottom: 1px solid var(--border);
}
.sb-eval-label { font-size: 0.72rem; color: var(--text-2); font-weight: 500; }
.sb-eval-score { font-size: 0.76rem; color: var(--accent); font-weight: 700; }
.sb-metric-row { display: flex; justify-content: space-between; padding: 0.15rem 0; }
.sb-metric-name { font-size: 0.69rem; color: var(--text-3); }
.sb-metric-val  { font-size: 0.69rem; color: var(--text-2); font-weight: 600; }

/* ─── Sidebar download button ───────────────────────────────────────────────── */
[data-testid="stSidebar"] .stDownloadButton > button {
    background: var(--bg-2) !important;
    color: var(--text-2) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    font-size: 0.76rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: background 0.15s, color 0.15s !important;
}
[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: var(--border) !important;
    color: var(--text-1) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ─── Hero ──────────────────────────────────────────────────────────────────── */
.hero {
    padding: 3.5rem 0 2.8rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 3rem;
}
.hero-eyebrow {
    font-size: 0.63rem; font-weight: 600; font-family: var(--sans);
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 0.75rem;
}
.hero-title {
    font-family: var(--sans);
    font-size: 2.4rem; font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.03em; margin-bottom: 0.6rem;
    background: linear-gradient(90deg, #2563EB, #2DD4BF);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}
.hero-tagline {
    font-family: var(--sans);
    font-size: 1.02rem; font-style: normal; font-weight: 400;
    color: var(--text-2); line-height: 1.6;
    max-width: 520px; margin-bottom: 0;
}

/* ─── Section headers ───────────────────────────────────────────────────────── */
.sec-head {
    font-size: 0.62rem; font-weight: 600; font-family: var(--sans);
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-3); margin-bottom: 1.1rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--border);
}
.section-gap { margin-top: 2.8rem; }

/* ─── Upload zone — the REAL native uploader, fully restyled ─────────────────
   There is exactly one upload element on screen: Streamlit's own
   st.file_uploader. We turn its dropzone <section> into the large dashed
   accent box, recolor its own icon, swap its default copy for our own via
   ::before/::after, and restyle its own Browse button. No second/decorative
   element is rendered anywhere — that was the source of the duplicate UI. */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed var(--accent-md) !important;
    border-radius: var(--radius) !important;
    background: var(--accent-tint) !important;
    padding: 2.4rem 1.5rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.85rem !important;
    text-align: center !important;
}
[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-lt) !important;
}

/* Icon + primary-text wrapper — Streamlit renders this as a row by default
   (icon beside text); force it into its own centered column so the icon
   sits above the text instead of beside it. */
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.5rem !important;
    margin: 0 !important;
}

/* Native cloud icon — recolored + enlarged, pinned to the top of the stack */
[data-testid="stFileUploader"] section svg,
[data-testid="stFileUploaderDropzone"] svg {
    width: 34px !important; height: 34px !important;
    color: var(--accent) !important;
    margin: 0 !important;
    order: -1 !important;
}

/* Native instruction text + size-limit caption — hidden, replaced with our
   own copy via pseudo-elements */
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    display: none !important;
}
/* Primary text — second item in the icon/text wrapper, right under the icon */
[data-testid="stFileUploader"] section > div::before,
[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content: "Drag & drop your research papers here, or click to browse";
    display: block;
    order: 1;
    font-size: 0.88rem; font-weight: 600; line-height: 1.5;
    color: var(--text-1); font-family: var(--sans);
    max-width: 360px;
}
/* Hint text — attached to the OUTER section (not the icon/text wrapper) so
   it falls after the Browse button in source order, landing at the bottom */
[data-testid="stFileUploader"] section::after,
[data-testid="stFileUploaderDropzone"]::after {
    content: "PDF files only · AI, ML, NLP, Computer Vision";
    display: block;
    font-size: 0.74rem; color: var(--text-3); font-family: var(--sans);
}

/* Native Browse button — accent pill, second-to-last in the stack */
[data-testid="stFileUploader"] section button {
    background: var(--surface) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent-md) !important;
    border-radius: 8px !important;
    font-size: 0.77rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 1rem !important;
    margin: 0 !important;
    box-shadow: none !important;
    transform: none !important;
    transition: background 0.15s !important;
}
[data-testid="stFileUploader"] section button:hover {
    background: var(--accent-lt) !important;
}
[data-testid="stFileUploader"] > div { color: var(--text-3) !important; font-size: 0.74rem !important; }
[data-testid="stFileUploader"] label { display: none !important; }

/* ─── All buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.58rem 1.4rem !important;
    height: 42px !important;
    transition: filter 0.15s ease, transform 0.12s ease !important;
    box-shadow: 0 4px 14px 0 var(--accent-glow) !important;
}
.stButton > button:hover {
    background: var(--accent-dark) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 24px 0 var(--accent-glow) !important;
}
.stButton > button:active { transform: translateY(0) scale(0.99) !important; }

/* Ask button — right-rounded only, flush against input */
.prompt-btn-col .stButton > button {
    border-radius: 0 24px 24px 0 !important;
    height: 48px !important;
    padding: 0 1.5rem !important;
}

/* Force input + button onto the exact same baseline — collapsed labels can
   otherwise leave a few px of residual space above the widget */
.prompt-input-col, .prompt-btn-col {
    display: flex !important;
    align-items: flex-end !important;
}
.prompt-input-col [data-testid="stTextInput"],
.prompt-btn-col .stButton {
    width: 100% !important;
}
.prompt-input-col [data-testid="stWidgetLabel"],
.prompt-input-col [data-testid="stTextInput"] label {
    display: none !important;
    height: 0 !important; margin: 0 !important; padding: 0 !important;
}
.prompt-input-col [data-testid="stTextInput"] > div {
    margin: 0 !important;
}

/* ─── Text inputs ───────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    border: 1.5px solid var(--border-2) !important;
    border-radius: 12px !important;
    color: var(--text-1) !important;
    font-size: 0.9rem !important;
    padding: 0.72rem 1.1rem !important;
    height: 42px !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.04) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    font-family: var(--sans) !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.04), 0 0 0 3px var(--accent-md) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-3) !important; }
[data-testid="stTextInput"] label { color: var(--text-2) !important; font-size: 0.78rem !important; }

/* Prompt bar — left-rounded input connects to right-rounded button */
.prompt-input-col [data-testid="stTextInput"] input {
    border-right: none !important;
    border-radius: 24px 0 0 24px !important;
    height: 48px !important;
    font-size: 0.92rem !important;
    padding: 0.8rem 1.2rem !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-sm) !important;
}
.prompt-input-col [data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    border-right: none !important;
    box-shadow: var(--shadow-sm), 0 0 0 3px var(--accent-md) !important;
}

/* ─── Dividers ──────────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* ─── Question banner ───────────────────────────────────────────────────────── */
.q-banner {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1.6rem;
    font-size: 0.87rem;
    color: var(--text-2);
    font-family: var(--sans);
}
.q-banner strong { color: var(--text-1); }

/* ─── Comparison cards ──────────────────────────────────────────────────────── */
.answer-card {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 1.6rem 1.7rem 1.4rem;
    border: 1px solid var(--border);
    height: 100%;
    box-shadow: var(--shadow-md);
}
.answer-card.baseline { border-top: 2px solid var(--border-2); }
.answer-card.rag      { border-top: 2px solid var(--accent); }

.badge {
    display: inline-block; font-size: 0.59rem; font-weight: 600;
    padding: 2px 8px; border-radius: 99px; font-family: var(--sans);
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.65rem;
}
.badge-baseline {
    background: var(--bg-2); color: var(--text-2);
    border: 1px solid var(--border);
}
.badge-rag {
    background: var(--accent-lt); color: var(--accent);
    border: 1px solid var(--accent-md);
}

.card-title {
    font-family: var(--sans);
    font-size: 0.95rem; font-weight: 700;
    color: var(--text-1); margin-bottom: 0.2rem;
}
.card-sub {
    font-size: 0.72rem; color: var(--text-3);
    margin-bottom: 1.1rem; font-weight: 400;
    font-family: var(--sans);
}
.answer-body {
    font-size: 0.87rem; line-height: 1.82; color: var(--text-2);
    font-family: var(--sans);
}

.source-pill {
    display: inline-block;
    background: var(--accent-lt);
    color: var(--accent); font-size: 0.64rem; font-weight: 600;
    padding: 2px 8px; border-radius: 99px;
    margin: 6px 4px 0 0;
    border: 1px solid var(--accent-md);
    font-family: var(--sans);
}

/* VS column */
.vs-wrap { display: flex; align-items: center; justify-content: center; min-height: 160px; }
.vs-pill {
    background: var(--bg-2); color: var(--text-3);
    font-size: 0.59rem; font-weight: 700;
    padding: 7px 6px; border-radius: 99px;
    letter-spacing: 0.06em; border: 1px solid var(--border);
}

/* ─── Chunk card ────────────────────────────────────────────────────────────── */
.chunk-header {
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.84rem; font-weight: 600; color: var(--text-1);
    font-family: var(--sans); margin-bottom: 0.55rem;
}
.chunk-score {
    font-size: 0.68rem; font-weight: 600; color: var(--accent);
    background: var(--accent-lt); border: 1px solid var(--accent-md);
    border-radius: 99px; padding: 2px 9px; white-space: nowrap;
}
.chunk-card {
    background: var(--bg-2);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
    font-size: 0.82rem;
    color: var(--text-2);
    line-height: 1.72;
    font-family: var(--sans);
}

/* ─── Metric widget ─────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.9rem !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="metric-container"] label {
    color: var(--text-3) !important; font-size: 0.7rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text-1) !important; font-size: 1.05rem !important;
}

/* ─── Expanders ─────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 0.7rem !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-2) !important; font-size: 0.82rem !important;
    font-weight: 600 !important; padding: 0.8rem 1rem !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text-1) !important; }

/* ─── Alerts ─────────────────────────────────────────────────────────────────
   Streamlit's default alert palette (e.g. pale yellow warning text) was
   tuned for a dark theme. On our bright white background that text was
   nearly invisible. Each kind now gets an explicit soft background + a
   dark, readable text color, covering both the outer alert wrapper and the
   inner content box since Streamlit versions differ in which one paints
   the visible background. */
.stAlert {
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    overflow: hidden !important;
}

/* Warning — soft amber */
.stAlert:has([data-testid="stAlertContentWarning"]),
[data-testid="stAlertContentWarning"] {
    background: #FFFBEB !important;
    border: 1px solid rgba(180,83,9,0.18) !important;
    border-radius: 10px !important;
}
[data-testid="stAlertContentWarning"],
[data-testid="stAlertContentWarning"] p,
[data-testid="stAlertContentWarning"] span,
[data-testid="stAlertContentWarning"] svg {
    color: #B45309 !important;
}

/* Error — soft red */
.stAlert:has([data-testid="stAlertContentError"]),
[data-testid="stAlertContentError"] {
    background: #FEF2F2 !important;
    border: 1px solid rgba(185,28,28,0.18) !important;
    border-radius: 10px !important;
}
[data-testid="stAlertContentError"],
[data-testid="stAlertContentError"] p,
[data-testid="stAlertContentError"] span,
[data-testid="stAlertContentError"] svg {
    color: #B91C1C !important;
}

/* Success — soft emerald */
.stAlert:has([data-testid="stAlertContentSuccess"]),
[data-testid="stAlertContentSuccess"] {
    background: #ECFDF5 !important;
    border: 1px solid rgba(4,120,87,0.18) !important;
    border-radius: 10px !important;
}
[data-testid="stAlertContentSuccess"],
[data-testid="stAlertContentSuccess"] p,
[data-testid="stAlertContentSuccess"] span,
[data-testid="stAlertContentSuccess"] svg {
    color: #047857 !important;
}

/* Info — soft blue, on-brand with the accent */
.stAlert:has([data-testid="stAlertContentInfo"]),
[data-testid="stAlertContentInfo"] {
    background: var(--accent-tint) !important;
    border: 1px solid var(--accent-md) !important;
    border-radius: 10px !important;
}
[data-testid="stAlertContentInfo"],
[data-testid="stAlertContentInfo"] p,
[data-testid="stAlertContentInfo"] span,
[data-testid="stAlertContentInfo"] svg {
    color: var(--accent-dark) !important;
}

/* ─── Progress bar ──────────────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent), #60A5FA) !important;
    border-radius: 99px !important;
}

/* ─── Caption ───────────────────────────────────────────────────────────────── */
.stCaption, small { color: var(--text-3) !important; font-size: 0.74rem !important; }

/* ─── Blockquote ────────────────────────────────────────────────────────────── */
blockquote {
    border-left: 3px solid var(--border-2) !important;
    color: var(--text-2) !important;
    background: var(--bg-2) !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 0.5rem 0.9rem !important;
    font-size: 0.81rem !important;
    font-style: italic !important;
}

/* ─── Slider ────────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] .st-emotion-cache-1n76uvr { background: var(--accent) !important; }

/* ─── Score boxes ───────────────────────────────────────────────────────────── */
.score-row { display: flex; gap: 0.65rem; flex-wrap: wrap; }
.metric-box {
    background: var(--accent-lt);
    border: 1px solid var(--accent-md);
    border-radius: 10px; padding: 0.5rem 0.9rem;
    font-size: 0.83rem; flex: 1; text-align: center; color: var(--text-2);
}

/* ─── Spacing helpers ───────────────────────────────────────────────────────── */
.mt-20 { margin-top: 1.25rem; }
.mt-32 { margin-top: 2rem; }

/* ─── Global legibility fix (light theme) ──────────────────────────────────────
   Catches any remaining markdown/caption/label/div text that may still be
   inheriting a light color from Streamlit's base theme instead of our own
   --text-1 / --text-2 / --text-3 tokens. Scoped with low-priority base rules
   plus explicit carve-outs so it never overrides accent colors, links, or
   the gradient hero title below. */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: var(--text-1);
}
[data-testid="stMarkdownContainer"] h1:not(.hero-title),
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: var(--text-1);
}

/* Carve-out: gradient hero title must stay clipped/transparent */
.hero-title, .hero-title * {
    color: transparent !important;
    -webkit-text-fill-color: transparent !important;
}

/* Carve-out: links keep the accent color, not slate */
[data-testid="stMarkdownContainer"] a,
.stApp a {
    color: var(--accent) !important;
}
[data-testid="stMarkdownContainer"] a:hover,
.stApp a:hover {
    color: var(--accent-dark) !important;
}

/* Carve-out: buttons keep their own (white) text */
.stButton button, .stButton button * {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
def _init_state():
    if "pipeline" not in st.session_state:
        llm = get_llm_from_config()
        if llm:
            pipeline = RAGPipeline(llm)
            pipeline.try_load_from_disk()
            st.session_state.pipeline = pipeline
        else:
            st.session_state.pipeline = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "eval_logger" not in st.session_state:
        st.session_state.eval_logger = EvaluationLogger()


_init_state()

pipeline: RAGPipeline | None = st.session_state.pipeline
logger: EvaluationLogger = st.session_state.eval_logger

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ResearchRAG")
    st.markdown("---")

    # LLM status — minimal dot + text, no boxes
    if pipeline:
        st.markdown(
            f'<div class="sb-status"><span class="sb-dot sb-dot-green"></span>'
            f'LLM: {pipeline.llm.model_name}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sb-status"><span class="sb-dot sb-dot-amber"></span>'
            'LLM not configured</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Knowledge Base")

    if pipeline and pipeline.is_ready:
        files = pipeline.indexed_files
        st.markdown(
            f'<div class="sb-status"><span class="sb-dot sb-dot-green"></span>'
            f'Index: Ready &middot; {pipeline.total_chunks} chunks &middot; {len(files)} file(s)</div>',
            unsafe_allow_html=True,
        )
        for fname in files:
            st.markdown(
                f'<div class="sb-file">'
                f'<svg width="11" height="11" viewBox="0 0 24 24" fill="none" '
                f'stroke="currentColor" stroke-width="2">'
                f'<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
                f'<polyline points="14 2 14 8 20 8"/></svg>{fname}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="sb-status"><span class="sb-dot sb-dot-amber"></span>'
            'Index: No documents indexed</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Evaluation")

    summary = logger.summary()
    if summary:
        for mode, metrics in summary.items():
            avg = metrics.get("average", 0)
            badge = format_score_badge(avg)
            st.markdown(
                f'<div class="sb-eval-row">'
                f'<span class="sb-eval-label">{mode.upper()}</span>'
                f'<span class="sb-eval-score">{avg}/10 {badge}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for m in ("relevance", "completeness", "clarity", "source_coverage"):
                st.markdown(
                    f'<div class="sb-metric-row">'
                    f'<span class="sb-metric-name">{m.capitalize()}</span>'
                    f'<span class="sb-metric-val">{metrics.get(m, 0)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No evaluations logged yet.")

    st.markdown(" ")
    if os.path.exists(cfg.EVALUATION_LOG_PATH):
        with open(cfg.EVALUATION_LOG_PATH, "rb") as f:
            st.download_button(
                "Download Evaluation CSV",
                data=f,
                file_name="evaluation_log.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Research Assistant</div>
    <div class="hero-title">ResearchRAG</div>
    <div class="hero-tagline">Turn stacks of academic papers into instant, verifiable answers.</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Document Upload
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">Document Upload</div>', unsafe_allow_html=True)

col_upload, col_status = st.columns([2, 1], gap="large")

with col_upload:
    # Single, real upload element — the native st.file_uploader, fully
    # restyled via CSS below to render as the dashed accent dropzone.
    # (No separate decorative div: that was rendering as a sibling block
    # above the native widget, producing the duplicate-uploader bug.)
    uploaded_files = st.file_uploader(
        "Upload academic PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supported topics: AI, ML, DL, NLP, Computer Vision",
        label_visibility="collapsed",
    )

with col_status:
    st.markdown('<div class="mt-20"></div>', unsafe_allow_html=True)
    if uploaded_files:
        st.markdown(
            f'<div class="sb-status"><span class="sb-dot sb-dot-green"></span>'
            f'{len(uploaded_files)} file(s) ready to index</div>',
            unsafe_allow_html=True,
        )
    if pipeline and pipeline.is_ready:
        st.markdown(
            f'<div class="sb-status"><span class="sb-dot sb-dot-green"></span>'
            f'Index ready &middot; {pipeline.total_chunks} chunks</div>',
            unsafe_allow_html=True,
        )

if uploaded_files:
    if st.button("Index Documents", type="primary", use_container_width=True):
        if not pipeline:
            st.error("LLM is not configured. Cannot process documents.")
        else:
            progress_bar = st.progress(0, text="Starting…")
            status_text = st.empty()

            def _progress(step: int, total: int, msg: str):
                pct = int((step / max(total, 1)) * 100)
                progress_bar.progress(pct, text=msg)
                status_text.markdown(f"**{msg}**")

            filepaths = save_uploaded_files(uploaded_files)
            try:
                n_chunks = pipeline.ingest_pdfs(
                    filepaths,
                    save_to_disk=True,
                    progress_callback=_progress,
                )
                progress_bar.progress(100, text="Done!")
                status_text.empty()
                st.success(
                    f"Processed {len(uploaded_files)} document(s) — "
                    f"{n_chunks} chunks indexed and cached."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Error during processing: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Query
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-head">Ask a Question</div>', unsafe_allow_html=True)

# Unified pill prompt bar: left-rounded input + right-rounded ask button
prompt_col, ask_col = st.columns([6, 1], gap="small")

with prompt_col:
    st.markdown('<div class="prompt-input-col">', unsafe_allow_html=True)
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What does this paper conclude about BERT fine-tuning?",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with ask_col:
    st.markdown('<div class="prompt-btn-col">', unsafe_allow_html=True)
    submit = st.button("Ask", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if submit:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not pipeline:
        st.error("LLM is not configured.")
    else:
        with st.spinner("Generating answers…"):
            try:
                result: QueryResult = pipeline.query(question)
                st.session_state.last_result = result
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                st.session_state.last_result = None

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Results
# ═══════════════════════════════════════════════════════════════════════════════
result: QueryResult | None = st.session_state.last_result

if result:
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">Results</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="q-banner"><strong>Question</strong>&nbsp;&nbsp;{result.question}</div>',
        unsafe_allow_html=True,
    )

    # ── Tabbed results interface ───────────────────────────────────────────────
    tab_baseline, tab_rag, tab_context, tab_evaluate = st.tabs(
        ["Baseline LLM Answer", "RAG Answer", "Retrieved Context", "Evaluate"]
    )

    # ── Tab 1: Baseline LLM Answer ─────────────────────────────────────────────
    with tab_baseline:
        st.markdown(
            '<div class="answer-card baseline">'
            '<span class="badge badge-baseline">Without Retrieval</span>'
            '<div class="card-title">Baseline LLM</div>'
            '<div class="card-sub">Parametric knowledge only — no documents consulted.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="answer-body">{result.baseline_answer}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Tab 2: RAG Answer ──────────────────────────────────────────────────────
    with tab_rag:
        st.markdown(
            '<div class="answer-card rag">'
            '<span class="badge badge-rag">With Retrieval</span>'
            '<div class="card-title">ResearchRAG</div>'
            '<div class="card-sub">Document-grounded answer via FAISS cosine retrieval.</div>',
            unsafe_allow_html=True,
        )
        if not pipeline or not pipeline.is_ready:
            st.warning("No documents indexed — RAG answer falls back to baseline.")
        pills = ""
        if result.retrieved_chunks:
            seen: list[str] = []
            for rc in result.retrieved_chunks:
                if rc.source_label not in seen:
                    seen.append(rc.source_label)
                    pills += f'<span class="source-pill">{rc.source_label}</span>'
        st.markdown(
            f'<div class="answer-body">{result.rag_answer}</div>{pills}</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 3: Retrieved Context (+ nested Debug expander) ────────────────────
    with tab_context:
        st.markdown("### Retrieved Chunks")
        if not result.retrieved_chunks:
            st.info("No chunks retrieved. Upload and process documents first.")
        else:
            st.caption(
                f"Top {len(result.retrieved_chunks)} chunks retrieved "
                f"(cosine similarity, FAISS IndexFlatIP)"
            )
            for i, rc in enumerate(result.retrieved_chunks, 1):
                st.markdown(
                    f'<div class="chunk-header">'
                    f'<span>Chunk {i} — {rc.source_label}</span>'
                    f'<span class="chunk-score">score {rc.score:.4f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="chunk-card">{rc.chunk.text}</div>',
                    unsafe_allow_html=True,
                )
                col1, col2, col3 = st.columns(3)
                col1.metric("Similarity", f"{rc.score:.4f}")
                col2.metric("Tokens (est.)", rc.chunk.token_estimate)
                col3.metric("Page", rc.chunk.page_number)
                if i < len(result.retrieved_chunks):
                    st.divider()

        # Debug — Raw Retrieved Context, consolidated inside this tab
        with st.expander("Debug: Raw Retrieved Context", expanded=False):
            st.caption(
                "Exact text passed to the LLM as context (result.context_used), "
                "verbatim — useful for diagnosing retrieval quality or prompt issues."
            )
            if result.context_used:
                st.code(result.context_used, language=None)
            else:
                st.info("No context was used for this query (no chunks retrieved).")

    # ── Tab 4: Evaluate ─────────────────────────────────────────────────────────
    with tab_evaluate:
        st.markdown("### Manual Evaluation")
        st.caption(
            "Score each answer on the four metrics below (1 = poor, 10 = excellent). "
            "Results are saved to CSV."
        )
        for mode, answer in [("baseline", result.baseline_answer), ("rag", result.rag_answer)]:
            st.markdown(f"#### {mode.upper()} Answer")
            st.markdown(f"> {truncate_text(answer, 300)}")

            cols = st.columns(4)
            scores = {}
            for col, metric in zip(cols, METRIC_NAMES):
                key_slug = metric.lower().replace(" ", "_")
                with col:
                    scores[key_slug] = st.slider(
                        metric,
                        min_value=1,
                        max_value=10,
                        value=5,
                        key=f"{mode}_{key_slug}_slider",
                        help=METRIC_DESCRIPTIONS[metric],
                    )

            notes = st.text_input(
                "Notes (optional)",
                key=f"{mode}_notes",
                placeholder="Any observations about this answer…",
            )

            if st.button(f"Save {mode.upper()} Score", key=f"save_{mode}"):
                ev = EvaluationScore(
                    question=result.question,
                    mode=mode,
                    answer=answer,
                    relevance=scores["relevance"],
                    completeness=scores["completeness"],
                    clarity=scores["clarity"],
                    source_coverage=scores["source_coverage"],
                    notes=notes,
                )
                logger.log(ev)
                st.success(
                    f"{mode.upper()} score saved — "
                    f"Average: {ev.average_score}/10 {format_score_badge(ev.average_score)}"
                )
            st.markdown("---")
