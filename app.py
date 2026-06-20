"""
app.py — ResearchRAG Streamlit application.

Layout
------
Sidebar  : LLM status, indexed file list, evaluation summary
Main     : Upload → Process → Query → Comparison Dashboard → Retrieval Analysis
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

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchRAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── General ── */
.main-title  { font-size: 2.2rem; font-weight: 700; color: #1a73e8; }
.sub-title   { font-size: 1.05rem; color: #555; margin-top: -0.5rem; }

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem 1.5rem 2.5rem;
    margin-bottom: 1.5rem;
    color: #fff;
}
.hero-title {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem 0;
}
.hero-sub {
    font-size: 0.95rem;
    color: #a8b9d4;
    margin: 0 0 1.2rem 0;
}

/* ── System stats strip ── */
.sys-banner {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}
.sys-stat {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 0.5rem 1.1rem;
    flex: 1;
    min-width: 110px;
    text-align: center;
}
.sys-stat-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #fff;
    display: block;
}
.sys-stat-label {
    font-size: 0.68rem;
    color: #a8b9d4;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ── Question banner ── */
.question-banner {
    background: #f0f4ff;
    border: 1px solid #c7d7f9;
    border-radius: 12px;
    padding: 0.85rem 1.25rem;
    margin-bottom: 1.25rem;
    font-size: 1rem;
    color: #1e3a8a;
    font-weight: 500;
}

/* ── Answer cards ── */
.answer-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.4rem 1.5rem 1.2rem 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
    border: 1px solid #e5e7eb;
    height: 100%;
    position: relative;
}
.answer-card.rag {
    border-top: 4px solid #7c3aed;
}
.answer-card.baseline {
    border-top: 4px solid #d97706;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.badge-baseline { background: #fef3c7; color: #92400e; }
.badge-rag      { background: #ede9fe; color: #5b21b6; }

/* ── Card title ── */
.card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #111827;
    margin: 0 0 0.2rem 0;
}
.card-subtitle {
    font-size: 0.78rem;
    color: #6b7280;
    margin-bottom: 0.9rem;
}

/* ── Answer body ── */
.answer-body {
    font-size: 0.9rem;
    line-height: 1.7;
    color: #374151;
    padding-bottom: 1rem;
    border-bottom: 1px solid #f3f4f6;
    margin-bottom: 0.8rem;
}

/* ── Info source pill ── */
.info-source {
    font-size: 0.75rem;
    color: #6b7280;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    display: inline-block;
    padding: 3px 10px;
    margin-top: 0.1rem;
}

/* ── Source pills ── */
.source-pill {
    display: inline-block;
    background: #ede9fe;
    color: #5b21b6;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 20px;
    margin: 2px 3px 2px 0;
}

/* ── VS divider ── */
.vs-col {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 200px;
}
.vs-pill {
    background: #e5e7eb;
    color: #6b7280;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 8px 10px;
    border-radius: 99px;
    letter-spacing: 0.05em;
}

/* ── Stats strip below cards ── */
.stats-strip {
    display: flex;
    gap: 0.75rem;
    margin: 1.2rem 0 0 0;
    flex-wrap: wrap;
}
.stat-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.55rem 1rem;
    flex: 1;
    text-align: center;
    min-width: 100px;
}
.stat-box-val  { font-size: 1.1rem; font-weight: 700; color: #1e293b; display: block; }
.stat-box-lbl  { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Retrieval Analysis ── */
.ra-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e1b4b;
    margin: 1.75rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #ede9fe;
}
.ra-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #7c3aed;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.ra-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
    margin-bottom: 0.5rem;
}
.ra-rank {
    background: #7c3aed;
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}
.ra-tag {
    background: #f3f4f6;
    color: #374151;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 20px;
}
.ra-score-label {
    font-size: 0.72rem;
    color: #6b7280;
    margin-left: auto;
}
.ra-preview {
    font-size: 0.82rem;
    color: #4b5563;
    line-height: 1.55;
    font-style: italic;
    border-top: 1px solid #f3f4f6;
    padding-top: 0.45rem;
    margin-top: 0.3rem;
}

/* ── Citation block ── */
.citation-block { margin-top: 0.6rem; }
.citation-title { font-size: 0.72rem; font-weight: 700; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.citation-item  { font-size: 0.76rem; color: #374151; margin-bottom: 0.15rem; }
.citation-num   { background: #7c3aed; color: #fff; font-size: 0.62rem; font-weight: 700; padding: 1px 6px; border-radius: 99px; margin-right: 4px; }

/* ── chunk-card (kept for legacy context expander) ── */
.chunk-card {
    background: #f8f9fa;
    border-left: 4px solid #1a73e8;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.7rem;
    font-size: 0.88rem;
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
    st.markdown("## 🔬 ResearchRAG")
    st.markdown("---")

    if pipeline:
        st.success(f"✅ LLM: **{pipeline.llm.model_name}**")
    else:
        st.error("❌ LLM not configured. Check `.env` file.")

    st.markdown("---")
    st.markdown("### 📄 Indexed Documents")
    if pipeline and pipeline.is_ready:
        files = pipeline.indexed_files
        st.info(f"**{pipeline.total_chunks}** chunks from **{len(files)}** file(s)")
        for fname in files:
            st.markdown(f"- {fname}")
    else:
        st.warning("No documents indexed yet.")

    st.markdown("---")
    st.markdown("### 📊 Evaluation Summary")
    summary = logger.summary()
    if summary:
        for mode, metrics in summary.items():
            avg = metrics.get("average", 0)
            badge = format_score_badge(avg)
            st.markdown(f"**{mode.upper()}** {badge} avg: **{avg}/10**")
            for m in ("relevance", "completeness", "clarity", "source_coverage"):
                st.markdown(f" - {m.capitalize()}: {metrics.get(m, 0)}")
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


# ── Hero header with system stats ────────────────────────────────────────────
n_docs   = len(pipeline.indexed_files) if pipeline and pipeline.is_ready else 0
n_chunks = pipeline.total_chunks       if pipeline and pipeline.is_ready else 0
llm_name = pipeline.llm.model_name     if pipeline else "Not configured"

st.markdown(f"""
<div class="hero-header">
  <div class="hero-title">🔬 ResearchRAG</div>
  <div class="hero-sub">Knowledge-Grounded Academic Question Answering</div>
  <div class="sys-banner">
    <div class="sys-stat">
      <span class="sys-stat-value">{n_docs}</span>
      <span class="sys-stat-label">Documents</span>
    </div>
    <div class="sys-stat">
      <span class="sys-stat-value">{n_chunks}</span>
      <span class="sys-stat-label">Chunks</span>
    </div>
    <div class="sys-stat">
      <span class="sys-stat-value">FAISS</span>
      <span class="sys-stat-label">Retrieval</span>
    </div>
    <div class="sys-stat">
      <span class="sys-stat-value" style="font-size:0.85rem">{llm_name}</span>
      <span class="sys-stat-label">LLM</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


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
    st.markdown(" ")
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
            status_text  = st.empty()

            def _progress(step: int, total: int, msg: str):
                pct = int((step / max(total, 1)) * 100)
                progress_bar.progress(pct, text=msg)
                status_text.markdown(f"**{msg}**")

            filepaths = save_uploaded_files(uploaded_files)
            try:
                n = pipeline.ingest_pdfs(
                    filepaths,
                    save_to_disk=True,
                    progress_callback=_progress,
                )
                progress_bar.progress(100, text="Done!")
                status_text.empty()
                st.success(
                    f"✅ Processed **{len(uploaded_files)}** document(s) → "
                    f"**{n}** chunks indexed and cached."
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
# SECTION 3 — Comparison Dashboard + Retrieval Analysis
# ═══════════════════════════════════════════════════════════════════════════
result: QueryResult | None = st.session_state.last_result

if result:
    st.markdown("---")

    # Question banner
    st.markdown(
        f'<div class="question-banner">❓ {result.question}</div>',
        unsafe_allow_html=True,
    )

    # ── Side-by-side comparison ──────────────────────────────────────────
    left, mid, right = st.columns([11, 1, 11])

    # ── LEFT: Baseline ───────────────────────────────────────────────────
    with left:
        st.markdown("""
        <div class="answer-card baseline">
          <span class="badge badge-baseline">Without Retrieval</span>
          <div class="card-title">🤖 Baseline LLM</div>
          <div class="card-subtitle">LLM parametric knowledge only — no documents consulted</div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div class="answer-body">{result.baseline_answer}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="info-source">📚 Information source: General LLM Knowledge &nbsp;·&nbsp; No citations</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── MIDDLE: VS divider ───────────────────────────────────────────────
    with mid:
        st.markdown(
            '<div class="vs-col"><div class="vs-pill">VS</div></div>',
            unsafe_allow_html=True,
        )

    # ── RIGHT: RAG ───────────────────────────────────────────────────────
    with right:
        st.markdown("""
        <div class="answer-card rag">
          <span class="badge badge-rag">With Retrieval</span>
          <div class="card-title">🔬 ResearchRAG</div>
          <div class="card-subtitle">Document-grounded answer via FAISS retrieval</div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div class="answer-body">{result.rag_answer}</div>',
            unsafe_allow_html=True,
        )

        # Source pills (deduplicated, order-preserving)
        if result.retrieved_chunks:
            seen: list[str] = []
            pills_html = ""
            for rc in result.retrieved_chunks:
                lbl = rc.source_label
                if lbl not in seen:
                    seen.append(lbl)
                    pills_html += f'<span class="source-pill">📄 {lbl}</span>'

            # Numbered citation references
            citations_html = ""
            for idx, src in enumerate(seen, 1):
                citations_html += (
                    f'<div class="citation-item">'
                    f'<span class="citation-num">{idx}</span>{src}'
                    f'</div>'
                )

            st.markdown(f"""
            {pills_html}
            <div class="citation-block">
              <div class="citation-title">📎 Retrieved Document References</div>
              {citations_html}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Stats strip ──────────────────────────────────────────────────────
    n_retrieved  = len(result.retrieved_chunks) if result.retrieved_chunks else 0
    unique_srcs  = len({rc.source_label for rc in result.retrieved_chunks}) if result.retrieved_chunks else 0
    top_score    = f"{result.retrieved_chunks[0].score:.4f}" if result.retrieved_chunks else "—"
    avg_tokens   = (
        int(sum(rc.chunk.token_estimate for rc in result.retrieved_chunks) / n_retrieved)
        if n_retrieved else 0
    )

    st.markdown(f"""
    <div class="stats-strip">
      <div class="stat-box"><span class="stat-box-val">{n_retrieved}</span><span class="stat-box-lbl">Chunks Retrieved</span></div>
      <div class="stat-box"><span class="stat-box-val">{unique_srcs}</span><span class="stat-box-lbl">Source Files</span></div>
      <div class="stat-box"><span class="stat-box-val">{top_score}</span><span class="stat-box-lbl">Top Similarity</span></div>
      <div class="stat-box"><span class="stat-box-val">{avg_tokens}</span><span class="stat-box-lbl">Avg Tokens / Chunk</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Retrieval Analysis ───────────────────────────────────────────────
    if result.retrieved_chunks:
        st.markdown(
            '<div class="ra-header">🔍 Retrieval Analysis</div>',
            unsafe_allow_html=True,
        )

        # Sort descending by score (None-safe)
        sorted_chunks = sorted(
            result.retrieved_chunks,
            key=lambda c: c.score if c.score is not None else 0.0,
            reverse=True,
        )

        for rank, rc in enumerate(sorted_chunks, start=1):
            score    = rc.score
            filename = getattr(rc.chunk, "filename", rc.source_label)
            page     = getattr(rc.chunk, "page_number", "—")
            raw_text = (rc.chunk.text or "").strip()
            preview  = raw_text[:175] + "…" if len(raw_text) > 175 else raw_text

            # Progress-bar HTML for similarity score
            score_html = ""
            if score is not None:
                pct = min(int(score * 100), 100)
                score_html = f"""
                <div style="margin:0.35rem 0 0.15rem 0;">
                  <div style="display:flex;justify-content:space-between;
                              font-size:0.7rem;color:#6b7280;margin-bottom:3px;">
                    <span>Similarity score</span><span>{score:.4f}</span>
                  </div>
                  <div style="background:#e5e7eb;border-radius:99px;height:6px;">
                    <div style="width:{pct}%;
                                background:linear-gradient(90deg,#7c3aed,#a78bfa);
                                height:6px;border-radius:99px;"></div>
                  </div>
                </div>"""

            score_label = f'<span class="ra-score-label">Score: {score:.4f}</span>' if score is not None else ""

            st.markdown(f"""
            <div class="ra-card">
              <div class="ra-meta">
                <span class="ra-rank">#{rank}</span>
                <span class="ra-tag">📄 {filename}</span>
                <span class="ra-tag">p.&nbsp;{page}</span>
                {score_label}
              </div>
              {score_html}
              <div class="ra-preview">"{preview}"</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Expanders ────────────────────────────────────────────────────────
    with st.expander("📄 Full Retrieved Context", expanded=False):
        if not result.retrieved_chunks:
            st.info("No chunks retrieved. Upload and process documents first.")
        else:
            st.caption(
                f"Top **{len(result.retrieved_chunks)}** chunks "
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
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Similarity", f"{rc.score:.4f}")
                    c2.metric("Tokens (est.)", rc.chunk.token_estimate)
                    c3.metric("Page", rc.chunk.page_number)

    with st.expander("📊 Manual Evaluation", expanded=False):
        st.caption(
            "Score each answer on the four metrics below (1 = poor, 10 = excellent). "
            "Results are saved to CSV."
        )
        for mode, answer in [("baseline", result.baseline_answer), ("rag", result.rag_answer)]:
            st.markdown(f"#### {mode.upper()} Answer")
            st.markdown(f"> {truncate_text(answer, 300)}")

            cols   = st.columns(4)
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
