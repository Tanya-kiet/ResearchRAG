# 🔬 ResearchRAG

**Retrieval-Augmented Generation System for Academic Question Answering**
*with Comparative LLM Evaluation*

---

## 📌 Project Overview

ResearchRAG is a clean, modular RAG system designed for academic research evaluation.
It compares two answer-generation strategies side-by-side:

| Mode | Pipeline |
|------|----------|
| **Baseline LLM** | Question → LLM → Answer |
| **ResearchRAG** | Question → FAISS Retriever → PDF Chunks → LLM → Grounded Answer |

Built for clarity and reproducibility — suitable for academic submission and viva presentation.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       ResearchRAG System                      │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  PDF Loader  │───▶│   Chunker    │───▶│ EmbeddingModel │  │
│  │  (PyMuPDF)  │    │ (500–800 tok)│    │ (MiniLM-L6-v2) │  │
│  └─────────────┘    └──────────────┘    └───────┬────────┘  │
│                                                  │           │
│                                          ┌───────▼────────┐  │
│                                          │  FAISS Index   │  │
│                                          │ (IndexFlatIP)  │  │
│                                          └───────┬────────┘  │
│                                                  │           │
│  User Question ──── EmbeddingModel ─────────────▶│           │
│                                          ┌───────▼────────┐  │
│                                          │   Retriever    │  │
│                                          │ (cosine, top-K)│  │
│                                          └───────┬────────┘  │
│                                                  │           │
│                                          ┌───────▼────────┐  │
│                                          │   Generator    │  │
│                                          │ (RAG Prompt)   │  │
│                                          └───────┬────────┘  │
│                                                  │           │
│                                            RAG Answer        │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ResearchRAG/
│
├── app.py                 ← Streamlit UI (entry point)
├── config.py              ← Centralised configuration
│
├── rag/
│   ├── pipeline.py        ← End-to-end orchestrator
│   ├── retriever.py       ← FAISS-based chunk retrieval
│   ├── generator.py       ← LLM answer generation
│   └── prompt_templates.py← Baseline + RAG prompts
│
├── llm/
│   ├── base.py            ← Abstract LLM interface
│   ├── openai_llm.py      ← OpenAI ChatCompletion backend
│   └── claude_llm.py      ← Anthropic Claude backend
│
├── data/
│   ├── pdf_loader.py      ← PDF text extraction (PyMuPDF)
│   └── chunker.py         ← Overlapping text chunking
│
├── vectorstore/
│   ├── embeddings.py      ← SentenceTransformer wrapper
│   └── faiss_store.py     ← FAISS index + persistence
│
├── evaluation/
│   ├── metrics.py         ← Manual scoring schema (1–10)
│   └── logger.py          ← CSV + session logging
│
├── utils/
│   ├── helpers.py         ← Streamlit helpers
│   └── text_cleaning.py   ← Text normalisation
│
├── storage/
│   ├── faiss_index/       ← Persisted FAISS index + metadata
│   └── documents/         ← Copies of uploaded PDFs
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone / unzip the project

```bash
unzip ResearchRAG.zip
cd ResearchRAG
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Or, to use Claude:

```
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run the application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🔄 Example Workflow

1. **Upload PDFs** — drag and drop 15–20 academic papers (AI/ML/NLP/CV topics).
2. **Process Documents** — click the button; chunks are embedded and indexed in FAISS.
3. **Ask a question** — e.g. *"Explain the attention mechanism"*
4. **Compare answers** across three tabs:
   - `Baseline LLM Answer` — model's parametric knowledge only
   - `RAG Answer` — grounded in your documents
   - `Retrieved Context` — the exact chunks that informed the RAG answer
5. **Evaluate** — score each answer 1–10 on four metrics; download the CSV.

---

## 🧩 How FAISS Works

FAISS (Facebook AI Similarity Search) provides efficient approximate nearest-neighbour search over dense vector spaces.

- **Index type used:** `IndexFlatIP` (flat inner-product)
- **Why inner-product?** When embeddings are L2-normalised (unit vectors), inner product equals cosine similarity.
- **Search complexity:** O(n·d) for a flat index — exact, no approximation error.
- **Persistence:** The index is serialised to disk (`storage/faiss_index/index.faiss`) after the first ingestion run. On subsequent app starts it is reloaded — **no recomputation needed**.

---

## ✂️ How Chunking Works

Large PDF pages are split into overlapping windows to balance context and precision:

| Parameter | Default | Range |
|-----------|---------|-------|
| Chunk size | 700 tokens | 500–800 |
| Overlap | 100 tokens | 10–20 % |

**Algorithm:**
1. Extract text per page (PyMuPDF).
2. Slide a character window of size `chunk_size × 4` chars over the text.
3. Snap each window end to the nearest sentence boundary (`.!?` + whitespace) within a 200-char lookback window.
4. Advance by `(chunk_size – overlap) × 4` chars.
5. Each chunk stores: `chunk_id`, `filename`, `page_number`, `chunk_index`, `text`.

---

## 📊 Evaluation (Manual Only)

For each question, the evaluator scores both the Baseline and RAG answers on:

| Metric | Description | Scale |
|--------|-------------|-------|
| **Relevance** | How well the answer addresses the question | 1–10 |
| **Completeness** | Whether all key aspects are covered | 1–10 |
| **Clarity** | Ease of understanding | 1–10 |
| **Source Coverage** | Use of source document material (RAG) | 1–10 |

Scores are appended to `storage/evaluation_log.csv` with a timestamp.
A running average per mode is shown in the sidebar.

> ⚠️ **No automated fact-checking or hallucination detection is included.**
> This is intentional — the evaluation is purely manual and human-driven.

---

## 🤖 LLM Support

| Provider | Config value | Requires |
|----------|-------------|---------|
| OpenAI (default) | `LLM_PROVIDER=openai` | `OPENAI_API_KEY` |
| Anthropic Claude | `LLM_PROVIDER=claude` | `ANTHROPIC_API_KEY` |

Switching providers requires only a `.env` change and app restart.
Both backends implement the same `BaseLLM` interface, so the pipeline is provider-agnostic.

---

## ⚡ Performance Notes

- **Embedding model:** `all-MiniLM-L6-v2` (22 M params, ~80 ms / batch on CPU)
- **Index build time:** < 5 s for 20 PDFs (~2000 chunks)
- **Query latency:** < 1 s retrieval on CPU for datasets up to ~50 k chunks
- **Caching:** Embeddings and FAISS index are persisted; re-upload triggers rebuild only when explicitly requested.

---

## 📋 RAG Prompt

```
You are an expert academic assistant. Your task is to answer a question
using ONLY the context provided below from academic research papers.

Instructions:
- Use only information from the provided context.
- Do NOT use any external or prior knowledge.
- If the answer is not present in the context, respond with exactly:
  "Information not available in the provided documents."
- Be concise and precise.
- Cite the source filename when possible.

---
CONTEXT:
{context}
---

Question: {question}

Answer:
```

---

## ✅ Success Criteria

- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `streamlit run app.py` launches the UI
- [ ] PDFs are loaded, chunked, and indexed
- [ ] FAISS retrieval returns relevant chunks in < 1 s
- [ ] RAG answers are visibly grounded in document content
- [ ] Evaluation scores are saved to CSV
- [ ] Index persists across app restarts (no recomputation)

---

*ResearchRAG — Built for academic reproducibility and viva-ready clarity.*
