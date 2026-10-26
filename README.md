# DocQuery RAG — Document Q&A with cited answers

**RAG-Based Document Q&A System** · HCLTech Career Shaper Programme 2025-26 · Project No. 72 · Domain: LLM / Python

Ask questions about documents (built-in corpus of Indian public documents, or your own PDF / Word / TXT uploads) and get answers generated **only** from retrieved passages, with every claim cited back to its source chunk.

## Features

- **Ask** — chat-style Q&A. Hybrid retrieval (BM25 + ChromaDB vectors), grounded prompt, LLM answer with `[n]` citations, source chunks shown beside every answer.
- **Upload** — PDF, Word (.docx), TXT, Markdown, HTML. Chunked with LangChain's text splitter and indexed instantly.
- **LLM providers** — Groq (free tier), Grok (xAI), OpenAI or local Ollama, through one OpenAI-compatible client. No key? An offline extractive mode still answers with citations.
- **Evaluation** — EDA of `data/dataset.csv` (4 charts, key findings) and a live retriever benchmark (Hit@1, Hit@3, MRR, latency).
- **Architecture** — pipeline, level-0 DFD, tech stack and the exact prompt template.

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Vector store | ChromaDB (all-MiniLM-L6-v2 embeddings) + BM25 (`rank-bm25`) |
| LLM | Groq / Grok / OpenAI / Ollama via the `openai` SDK |
| Parsing | pypdf, python-docx |
| Charts | Plotly |

## Project structure

```
app.py                 Streamlit app (Ask, Documents, Evaluation, Architecture)
rag/engine.py          RAG pipeline: ingestion, hybrid index, prompt, generation, extractive fallback, benchmark
data/corpus.json       Built-in corpus (14 public-domain documents) + 16-question benchmark
data/dataset.csv       Project dataset (500 rows x 7 columns)
data/dataset_cleaned.csv, data/data_dictionary.csv   Cleaned dataset and column descriptions
.streamlit/config.toml Dark theme
.streamlit/secrets.toml.example   Template for the API key
```

---

## Option A — Deploy free on Streamlit Community Cloud (public link)

You need: a GitHub account and an API key (Groq is free; Grok/xAI needs credits).

1. **Get the code into your GitHub.** Open this repository on GitHub and click **Fork** (top right) → **Create fork**.
2. **Get an API key.**
   - Groq (free): go to <https://console.groq.com> → sign in → **API Keys** → **Create API Key** → copy it (starts with `gsk_`).
   - or Grok: <https://console.x.ai> → **API Keys** → create → copy it (starts with `xai-`).
3. **Create the app.** Go to <https://share.streamlit.io> → sign in with GitHub → **Create app** → **Deploy a public app from GitHub**.
   - Repository: `<your-username>/cspace_app`
   - Branch: `main`
   - Main file path: `app.py`
   - Optional: pick a custom URL, e.g. `docquery-rag`.
4. **Paste the key as a secret.** Before deploying, click **Advanced settings**:
   - Python version: **3.11** or **3.12**
   - In **Secrets**, paste one line (use the one for your key):
     ```toml
     GROQ_API_KEY = "gsk_paste_your_key_here"
     ```
     or
     ```toml
     XAI_API_KEY = "xai-paste_your_key_here"
     ```
   - Click **Save**.
5. Click **Deploy**. The first build takes about 3–5 minutes. You get a link like `https://docquery-rag.streamlit.app`.
6. Open the app. In the sidebar, choose the provider that matches your key (**Groq (free tier)** or **Grok (xAI)**). It shows *"API key loaded from app secrets"*. Ask a question; the grey label under the answer shows the model that wrote it.

To change the key later: app page → **⋮** → **Settings** → **Secrets**.
Free apps sleep after a few days without visits; the first visit then takes ~30 seconds to wake.

## Option B — Run on your own laptop

Requires Python 3.10–3.12 and Git.

```bash
git clone https://github.com/SARPAT/cspace_app.git
cd cspace_app
python -m venv .venv
# Windows:    .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. Paste your Groq or Grok key in the sidebar (it is kept only for that session), or save it permanently:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit the key inside
```

`secrets.toml` is in `.gitignore`, so the key is never committed.

## Troubleshooting

| Problem | Fix |
|---|---|
| Answers say *Extractive · offline* | No key found. Add it in Secrets (cloud) or paste it in the sidebar, and pick the matching provider. |
| *The LLM call failed (AuthenticationError…)* | The key is wrong or for a different provider. Check the provider dropdown. |
| *model not found* | Model names change. Type a current one in the **Model** box (Groq: `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`; xAI: `grok-3-mini`, `grok-4`). |
| Build fails on Streamlit Cloud mentioning sqlite | Make sure Python 3.11/3.12 is selected; `pysqlite3-binary` in requirements handles the sqlite version. |
| Sidebar says *BM25 only* | The embedding model could not be downloaded; search still works with keyword ranking. |
| A PDF gives no text | It is a scanned image. Run OCR first or upload a text-based PDF. |

## Ethics

RAG systems can still hallucinate. This app always shows the source chunks beside each answer, instructs the model to use only the retrieved context, and reports *"I could not find this in the documents"* when the answer is missing. Not for legal or medical advice without expert review. API keys live only in server secrets or the user's session; uploads stay in memory and are not stored.
