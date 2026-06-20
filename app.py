"""
app.py — ResearchRAG Streamlit application.

Layout
------
Sidebar  : LLM status, indexed file list, evaluation summary
Main     : Upload → Process → Query → Results (tabbed)
"""

from __future__ import annotations

import os

import streamlit as st

from config import cfg
from evaluation.logger import EvaluationLogger
from evaluation.metrics import EvaluationScore, METRIC_NAMES, METRIC_DESCRIPTIONS
from rag.pipeline import RAGPipeline, QueryResult
from utils.helpers import get_llm_from_config, save_uploaded_files, format_score_badge
from utils.text_cleaning import truncate_text

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchRAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a73e8; }
    .sub-title  { font-size: 1.05rem; color: #555; margin-top: -0.5rem; }
    .chunk-card {
        background: #f8f9fa;
        border-left: 4px solid #1a73e8;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.7rem;
        font-size: 0.88rem;
    }
    .score-row  { display: flex; gap: 1rem; flex-wrap: wrap; }
    .metric-box {
        background: #e8f0fe;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        flex: 1;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── session state ────────────────────────────────────────────────────────────
def _init_state():
    if "pipeline" not in st.session_state:
        llm = get_llm_from_config()
        if llm:
            pipeline = RAGPipeline(llm)
            # Try to restore cached index
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

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 ResearchRAG")
    st.markdown("---")

    # LLM status
    if pipeline:
        st.success(f"✅ LLM: **{pipeline.llm.model_name}**")
    else:
        st.error("❌ LLM not configured. Check `.env` file.")

    st.markdown("---")

    # Indexed documents
    st.markdown("### 📄 Indexed Documents")
    if pipeline and pipeline.is_ready:
        files = pipeline.indexed_files
        st.info(f"**{pipeline.total_chunks}** chunks from **{len(files)}** file(s)")
        for fname in files:
            st.markdown(f"- {fname}")
    else:
        st.warning("No documents indexed yet.")

    st.markdown("---")

    # Evaluation summary
    st.markdown("### 📊 Evaluation Summary")
    summary = logger.summary()
    if summary:
        for mode, metrics in summary.items():
            avg = metrics.get("average", 0)
            badge = format_score_badge(avg)
            st.markdown(f"**{mode.upper()}** {badge} avg: **{avg}/10**")
            for m in ("relevance", "completeness", "clarity", "source_coverage"):
                st.markdown(f"  - {m.capitalize()}: {metrics.get(m, 0)}")
    else:
        st.caption("No evaluations logged yet.")

    if os.path.exists(cfg.EVALUATION_LOG_PATH):
        with open(cfg.EVALUATION_LOG_PATH, "rb") as f:
            st.download_button(
                "⬇️ Download Evaluation CSV",
                data=f,
                file_name="evaluation_log.csv",
                mime="text/csv",
            )

# ── main content ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🔬 ResearchRAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Retrieval-Augmented Generation for Academic Question Answering</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Document Upload
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## 📁 Document Upload")
col_upload, col_status = st.columns([2, 1])

with col_upload:
    uploaded_files = st.file_uploader(
        "Upload academic PDF files (15–20 recommended)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supported topics: AI, ML, DL, NLP, Computer Vision",
    )

with col_status:
    st.markdown(" ")  # spacer
    if uploaded_files:
        st.info(f"**{len(uploaded_files)}** file(s) ready to process")
    if pipeline and pipeline.is_ready:
        st.success(f"Index ready: **{pipeline.total_chunks}** chunks")

if uploaded_files:
    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
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
                    f"✅ Processed **{len(uploaded_files)}** document(s) → "
                    f"**{n_chunks}** chunks indexed and cached."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Error during processing: {exc}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Query
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## 💬 Ask a Question")

query_col, btn_col = st.columns([5, 1])
with query_col:
    question = st.text_input(
        "Enter your academic question",
        placeholder="e.g. Explain the attention mechanism in transformer models",
        label_visibility="collapsed",
    )
with btn_col:
    submit = st.button("🔍 Ask", type="primary", use_container_width=True)

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

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Results
# ═══════════════════════════════════════════════════════════════════════════
result: QueryResult | None = st.session_state.last_result

if result:
    st.markdown("---")
    st.markdown("## 📋 Results")
    st.markdown(f"**Question:** {result.question}")

    tab_baseline, tab_rag, tab_context, tab_eval = st.tabs(
        ["🤖 Baseline LLM Answer", "🔬 RAG Answer", "📄 Retrieved Context", "📊 Evaluate"]
    )

    # ── Tab 1: Baseline ──────────────────────────────────────────────────────
    with tab_baseline:
        st.markdown("### 🤖 Baseline LLM Answer")
        st.caption("Answer generated using only the LLM's parametric knowledge — no documents consulted.")
        st.markdown(result.baseline_answer)

    # ── Tab 2: RAG answer ────────────────────────────────────────────────────
    with tab_rag:
        st.markdown("### 🔬 RAG Answer")
        st.caption("Answer grounded in your uploaded documents via FAISS retrieval.")
        if not pipeline or not pipeline.is_ready:
            st.warning("⚠️ No documents indexed. RAG answer falls back to baseline.")
        st.markdown(result.rag_answer)

        if result.retrieved_chunks:
            st.markdown(f"**Sources used ({len(result.retrieved_chunks)}):**")
            sources = sorted(
                set(rc.source_label for rc in result.retrieved_chunks)
            )
            for s in sources:
                st.markdown(f"  - 📄 {s}")

    # ── Tab 3: Context ───────────────────────────────────────────────────────
    with tab_context:
        st.markdown("### 📄 Retrieved Chunks")
        if not result.retrieved_chunks:
            st.info("No chunks retrieved. Upload and process documents first.")
        else:
            st.caption(
                f"Top **{len(result.retrieved_chunks)}** chunks retrieved "
                f"(cosine similarity, FAISS IndexFlatIP)"
            )
            for i, rc in enumerate(result.retrieved_chunks, 1):
                with st.expander(
                    f"Chunk {i} — {rc.source_label} | score: {rc.score:.4f}",
                    expanded=(i == 1),
                ):
                    st.markdown(
                        f'<div class="chunk-card">{rc.chunk.text}</div>',
                        unsafe_allow_html=True,
                    )
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Similarity", f"{rc.score:.4f}")
                    col2.metric("Tokens (est.)", rc.chunk.token_estimate)
                    col3.metric("Page", rc.chunk.page_number)

    # ── Tab 4: Evaluation ────────────────────────────────────────────────────
    with tab_eval:
        st.markdown("### 📊 Manual Evaluation")
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

            if st.button(f"💾 Save {mode.upper()} Score", key=f"save_{mode}"):
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
                    f"✅ {mode.upper()} score saved! "
                    f"Average: **{ev.average_score}/10** {format_score_badge(ev.average_score)}"
                )

            st.markdown("---")
