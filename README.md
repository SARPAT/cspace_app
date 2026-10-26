# DocQuery RAG — Document Q&A with cited answers

**RAG-Based Document Q&A System** · HCLTech Career Shaper Programme 2025-26 · Project No. 72 · Domain: LLM / Python

Ask questions about documents (built-in corpus of Indian public documents, or your own PDF / Word / TXT uploads) and get answers generated **only** from retrieved passages, with every claim cited back to its source chunk.

![Ask view](docs/screenshot-ask.png)

## Features

- **Ask** — chat-style Q&A. Hybrid retrieval (BM25 + ChromaDB vectors), grounded prompt, LLM answer with `[n]` citations, source chunks shown beside every answer.
- **Upload** — PDF, Word (.docx), TXT, Markdown, HTML. Chunked with LangChain's text splitter and indexed instantly.
- **LLM providers** — Groq (free tier), Grok (xAI), OpenAI or local Ollama, through one OpenAI-compatible client. No key? An offline extractive mode still answers with citations.
- **Evaluation** — EDA of `data/dataset.csv` (4 charts, key findings) and a live retriever benchmark (Hit@1, Hit@3, MRR, latency).
- **Architecture** — pipeline, level-0 DFD, tech stack and the exact prompt template.

![Evaluation view](docs/screenshot-evaluation.png)

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

## Getting started

Requires Python 3.10–3.12.

```bash
git clone https://github.com/SARPAT/cspace_app.git
cd cspace_app
python -m venv .venv
# Windows:    .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Configuration

Choose an LLM provider in the sidebar and paste its API key (kept only for that session), or store it in `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-key"     # Groq
# XAI_API_KEY = "your-key"    # Grok (xAI)
# OPENAI_API_KEY = "your-key" # OpenAI
```

A template is in `.streamlit/secrets.toml.example`; `secrets.toml` is git-ignored. Without a key the app answers in offline extractive mode. The model name can be changed in the sidebar.

## Ethics

RAG systems can still hallucinate. This app always shows the source chunks beside each answer, instructs the model to use only the retrieved context, and reports *"I could not find this in the documents"* when the answer is missing. Not for legal or medical advice without expert review. API keys live only in server secrets or the user's session; uploads stay in memory and are not stored.
