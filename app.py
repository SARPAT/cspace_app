"""DocQuery RAG — Streamlit app for the RAG-Based Document Q&A System (HCLTech Project 72)."""
from __future__ import annotations

import html
import re
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag.engine import (
    DATA_DIR,
    PROVIDERS,
    Document,
    HybridIndex,
    build_prompt,
    chunk_text,
    extract_text,
    extractive_answer,
    load_builtin,
    run_benchmark,
    stream_answer,
)

st.set_page_config(page_title="DocQuery RAG", page_icon="🔎", layout="wide")

# --------------------------------------------------------------------------- styles
TEAL, SAFFRON, SKY, ROSE, LIME = "#2EE6C1", "#FF9F45", "#62A8FF", "#FF6F91", "#B9E769"
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@600;800&family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton button {{ font-family: 'Instrument Sans', system-ui, sans-serif; }}
.stApp {{ background:
  radial-gradient(900px 520px at 8% -8%, rgba(255,159,69,.13), transparent 60%),
  radial-gradient(820px 560px at 100% 0%, rgba(46,230,193,.11), transparent 62%), #070B14; }}
[data-testid="stSidebar"] {{ background: #0B1120; border-right: 1px solid rgba(148,170,210,.14); }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
.hero {{ text-align:center; padding: 6px 0 4px; }}
.eyebrow {{ font: 500 11px/1.3 'JetBrains Mono', monospace; letter-spacing: .12em; text-transform: uppercase; color: {SAFFRON}; }}
.hero h1 {{ font: 800 clamp(28px,4.4vw,52px)/1.06 'Unbounded', sans-serif; letter-spacing: -.035em; margin: 10px 0 0; color: #EAF0FA; }}
.grad {{ background: linear-gradient(92deg,{SAFFRON} 0%,#FFD29A 30%,{TEAL} 70%,{SKY} 100%); -webkit-background-clip:text; background-clip:text; color: transparent; }}
.lede {{ color:#8E9AB2; max-width: 680px; margin: 12px auto 0; font-size: 16.5px; }}
.stats {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); max-width: 860px; margin: 22px auto 8px;
  border:1px solid rgba(148,170,210,.14); border-radius: 16px; background: rgba(255,255,255,.02); }}
.stats div {{ padding: 12px; text-align:center; border-right: 1px solid rgba(148,170,210,.14); }}
.stats div:last-child {{ border-right: 0; }}
.stats b {{ display:block; font: 700 21px/1.15 'Unbounded', sans-serif; color:#EAF0FA; }}
.stats span {{ font-size: 12px; color:#8E9AB2; }}
.cite {{ font: 600 11px/1 'JetBrains Mono', monospace; background: rgba(46,230,193,.15); color: {TEAL};
  border: 1px solid rgba(46,230,193,.35); border-radius: 6px; padding: 2px 6px; margin: 0 2px; vertical-align: 2px; }}
.src {{ border:1px solid rgba(148,170,210,.14); background: rgba(18,26,44,.72); border-radius: 14px; padding: 14px 16px; margin-bottom: 10px; }}
.src .h {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom: 6px; }}
.src .h b {{ color:#EAF0FA; font-size: 14.5px; }}
.src .n {{ font: 700 12px 'JetBrains Mono', monospace; color:#04221C; background:{TEAL}; border-radius: 6px; padding: 3px 7px; }}
.src .m {{ font: 11px 'JetBrains Mono', monospace; color:#5D6982; }}
.src p {{ margin: 0; color:#C3CDDE; font-size: 14px; line-height: 1.6; }}
.src mark {{ background: rgba(255,159,69,.22); color: #FFE0BF; border-radius: 4px; padding: 0 3px; }}
.bar {{ display:inline-block; width: 90px; height: 6px; border-radius: 3px; background: rgba(148,170,210,.15); vertical-align: middle; overflow:hidden; }}
.bar i {{ display:block; height:100%; background: linear-gradient(90deg,{TEAL},{SKY}); }}
.tag {{ font: 600 9.5px/1 'JetBrains Mono', monospace; letter-spacing:.08em; text-transform: uppercase; padding: 4px 7px; border-radius: 6px;
  color:{SKY}; background: rgba(98,168,255,.12); border: 1px solid rgba(98,168,255,.25); }}
.pill {{ display:inline-flex; gap:6px; align-items:center; font: 500 11.5px/1 'JetBrains Mono', monospace; padding: 6px 10px; border-radius: 999px;
  border:1px solid rgba(148,170,210,.14); color:#8E9AB2; background: rgba(255,255,255,.04); margin: 8px 6px 0 0; }}
.pill.ok {{ color:#4FE3A0; border-color: rgba(79,227,160,.3); background: rgba(79,227,160,.07); }}
.pill.warn {{ color:#FFC37A; border-color: rgba(255,195,122,.3); background: rgba(255,159,69,.10); }}
.ethic {{ margin-top: 12px; font-size: 12.5px; color:#FFC37A; background: rgba(255,159,69,.10); border: 1px solid rgba(255,159,69,.2); border-radius: 10px; padding: 8px 12px; }}
.node {{ border:1px solid rgba(148,170,210,.14); background: rgba(18,26,44,.72); border-radius: 14px; padding: 14px; height: 100%; }}
.node.ing {{ border-top: 2px solid {SAFFRON}; }} .node.qry {{ border-top: 2px solid {TEAL}; }}
.node .k {{ font: 600 10px 'JetBrains Mono', monospace; letter-spacing: .1em; text-transform: uppercase; color:#8E9AB2; }}
.node b {{ display:block; font: 700 14px/1.25 'Unbounded', sans-serif; margin: 8px 0 6px; color:#EAF0FA; }}
.node p {{ margin:0; font-size: 12.5px; color:#8E9AB2; }}
div[data-testid="stMetric"] {{ border:1px solid rgba(148,170,210,.14); background: rgba(18,26,44,.72); border-radius: 14px; padding: 12px 16px; }}
div[data-testid="stMetricValue"] {{ font-family: 'Unbounded', sans-serif; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
@media (max-width: 700px) {{ .stats {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- state
def get_secret(name: str | None) -> str:
    if not name:
        return ""
    try:
        return str(st.secrets.get(name, ""))
    except Exception:  # noqa: BLE001 - no secrets.toml locally
        return ""


@st.cache_data
def builtin_data():
    return load_builtin()


if "index" not in st.session_state:
    docs, bench = builtin_data()
    with st.spinner("Building the vector index (first start downloads a small embedding model)…"):
        idx = HybridIndex()
        idx.add([Document(d.doc_id, d.title, d.category, d.doc_type, list(d.chunks)) for d in docs])
    st.session_state.index = idx
    st.session_state.history = []
    st.session_state.uploaded = set()
    st.session_state.bench = run_benchmark(idx, bench)
index: HybridIndex = st.session_state.index
_, BENCH = builtin_data()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🔎 DocQuery RAG")
    st.caption("RAG-Based Document Q&A System · Project 72")
    st.markdown("**Answer engine**")
    provider = st.selectbox("LLM provider", list(PROVIDERS), index=0)
    cfg = PROVIDERS[provider]
    secret_key = get_secret(cfg["secret"])
    if cfg["secret"] is None:
        api_key = ""
        st.caption("Runs against Ollama on this computer: `ollama run llama3.2`.")
    elif secret_key:
        api_key = secret_key
        st.success("API key loaded from app secrets", icon="🔐")
    else:
        api_key = st.text_input(
            f"{provider.split(' (')[0]} API key",
            type="password",
            help="Pasted keys stay in this browser session only and are never saved.",
            placeholder="Paste your key",
        )
        if provider.startswith("Groq"):
            st.caption("Free key: console.groq.com → API Keys")
        elif provider.startswith("Grok"):
            st.caption("Key: console.x.ai → API Keys")
    model = st.text_input("Model", value=cfg["model"])
    llm_ready = bool(api_key) or cfg["secret"] is None
    if not llm_ready:
        st.info("No key: answers use the offline extractive mode.", icon="ℹ️")
    top_k = st.slider("Top-k chunks", 2, 8, 4)

    st.divider()
    st.markdown("**Add your documents**")
    files = st.file_uploader(
        "PDF, Word, TXT, Markdown or HTML", type=["pdf", "docx", "txt", "md", "html", "htm"], accept_multiple_files=True
    )
    for f in files or []:
        key = (f.name, f.size)
        if key in st.session_state.uploaded:
            continue
        text, dtype = extract_text(f.name, f.getvalue())
        chunks = chunk_text(text)
        st.session_state.uploaded.add(key)
        if not chunks:
            st.warning(f"No readable text in {f.name}. Scanned PDFs need OCR first.")
            continue
        n = sum(1 for d in index.docs if d.category == "Your uploads") + 1
        index.add([Document(f"UPL-{n:02d}", f.name.rsplit(".", 1)[0], "Your uploads", dtype, chunks)])
        st.toast(f"Indexed {f.name}: {len(chunks)} chunks", icon="✅")
    ups = [d for d in index.docs if d.category == "Your uploads"]
    if ups:
        st.caption("Indexed uploads: " + ", ".join(d.title for d in ups))
    st.divider()
    st.caption(
        "Search: BM25 + ChromaDB dense vectors" if index.dense_on else "Search: BM25 only (dense model unavailable)"
    )

# --------------------------------------------------------------------------- helpers
STOPWORDS = {"the", "and", "for", "what", "who", "how", "are", "does", "with", "under", "this", "that", "from"}


def highlight(text: str, question: str) -> str:
    terms = {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", question)} - STOPWORDS
    out = html.escape(text)
    for t in sorted(terms, key=len, reverse=True):
        out = re.sub(rf"(?i)\b({re.escape(t)}\w*)", r"<mark>\1</mark>", out)
    return out


def cite_html(md: str) -> str:
    return re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', md)


def render_sources(hits, question):
    for i, h in enumerate(hits, 1):
        c = h.chunk
        score = (
            '<span class="tag">Document opening</span>'
            if h.header
            else f'<span class="m">score {h.score:.3f}</span> <span class="bar"><i style="width:{min(100, h.score * 100):.0f}%"></i></span>'
        )
        st.markdown(
            f"""<div class="src"><div class="h"><span class="n">{i}</span><b>{html.escape(c.doc.title)}</b>
<span class="tag">{c.doc.doc_type}</span><span class="m">{c.chunk_id} · {c.doc.doc_id}</span>{score}</div>
<p>{highlight(c.text, question)}</p></div>""",
            unsafe_allow_html=True,
        )


def render_turn(turn):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(turn["q"])
    with st.chat_message("assistant", avatar="🔎"):
        st.markdown(cite_html(turn["answer"]), unsafe_allow_html=True)
        not_found = "could not find this" in turn["answer"].lower()
        n_hits = len(turn["hits"])
        status = "Not in documents" if not_found else f"Grounded in {n_hits} chunks"
        pills = [
            f'<span class="pill {"warn" if not_found else "ok"}">● {status}</span>',
            f'<span class="pill">✦ {html.escape(turn["engine"])}</span>',
            f'<span class="pill">retrieval {turn["t_retrieve"]:.1f} ms</span>',
            f'<span class="pill">total {turn["t_total"]:.2f} s</span>',
        ]
        st.markdown("".join(pills), unsafe_allow_html=True)
        if turn.get("note"):
            st.caption(turn["note"])
        st.markdown(
            '<div class="ethic">⚠ Generated only from the cited source chunks. Check the sources; not legal or medical advice.</div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"Sources ({len(turn['hits'])})", expanded=turn is st.session_state.history[-1]):
            render_sources(turn["hits"], turn["q"])


def answer(question: str):
    t0 = time.perf_counter()
    base = index.retrieve(question, top_k)
    t_ret = (time.perf_counter() - t0) * 1000
    hits = index.with_headers(base) if base else []
    turn = {"q": question, "hits": hits, "t_retrieve": t_ret, "note": ""}
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🔎"):
        text = ""
        if not base:
            text, engine = "I could not find this in the documents.", "No relevant chunks"
        elif llm_ready:
            try:
                placeholder = st.empty()
                placeholder.markdown("_Reading the sources…_")
                for piece in stream_answer(provider, api_key, model, build_prompt(question, hits)):
                    text += piece
                    placeholder.markdown(cite_html(text) + " ▍", unsafe_allow_html=True)
                placeholder.empty()
                text, engine = text.strip(), f"{provider.split(' (')[0]} · {model}"
            except Exception as e:  # noqa: BLE001 - show the reason, fall back to extractive
                text = extractive_answer(index, question, hits)
                engine = "Extractive (fallback)"
                turn["note"] = f"The LLM call failed ({type(e).__name__}: {str(e)[:160]}). Showing the offline answer."
        else:
            text, engine = extractive_answer(index, question, hits), "Extractive · offline"
            turn["note"] = "Offline mode quotes matching passages. Add an API key in the sidebar for written answers."
    turn.update(answer=text, engine=engine, t_total=time.perf_counter() - t0)
    st.session_state.history.append(turn)
    st.rerun()


# --------------------------------------------------------------------------- header
b = st.session_state.bench
st.markdown(
    f"""<div class="hero"><div class="eyebrow">Retrieval-Augmented Generation · Document Q&amp;A</div>
<h1>Ask your documents.<br><span class="grad">Get answers with proof.</span></h1>
<p class="lede">Answers are generated only from retrieved passages, and every claim links back to its source chunk.</p>
<div class="stats"><div><b>{len(index.docs)}</b><span>documents indexed</span></div><div><b>{len(index.chunks)}</b><span>searchable chunks</span></div>
<div><b>{b['hit1'] * 100:.0f}%</b><span>top-1 accuracy</span></div><div><b>{b['latency_ms']:.1f} ms</b><span>retrieval latency</span></div></div></div>""",
    unsafe_allow_html=True,
)

tab_ask, tab_docs, tab_eval, tab_arch = st.tabs(["💬 Ask", "📚 Documents", "📊 Evaluation", "🧭 Architecture"])

# --------------------------------------------------------------------------- Ask
SAMPLES = [
    "How much money do farmers get under PM-KISAN?",
    "What does Article 21 protect?",
    "How many days does a PIO have to reply to an RTI request?",
    "Compare the insurance cover under PMJDY and PM-JAY.",
    "Which writs can the Supreme Court issue under Article 32?",
    "What is the capital of Australia?",
]
with tab_ask:
    pick = st.pills("Try a question", SAMPLES, selection_mode="single", key=f"pick{len(st.session_state.history)}")
    for turn in st.session_state.history:
        render_turn(turn)
    if not st.session_state.history:
        st.info("Ask about the built-in corpus (Constitution of India, government schemes, Acts), or upload your own PDF in the sidebar.", icon="💡")
    q = st.chat_input("Ask a question about the documents…")
    if pick:
        answer(pick)
    elif q:
        answer(q)

# --------------------------------------------------------------------------- Documents
with tab_docs:
    words = sum(c.words for c in index.chunks)
    st.markdown(f"#### Document corpus\n{len(index.docs)} documents · {len(index.chunks)} chunks · {words:,} words indexed. Upload more from the sidebar.")
    cats = list(dict.fromkeys(d.category for d in index.docs))
    for cat in cats:
        st.markdown(f"##### {cat}")
        cols = st.columns(2)
        for i, d in enumerate([d for d in index.docs if d.category == cat]):
            cs = [c for c in index.chunks if c.doc is d]
            with cols[i % 2].expander(f"{d.title}  ·  {d.doc_type}  ·  {len(cs)} chunks"):
                for c in cs:
                    st.markdown(f"<div class='src'><div class='h'><span class='m'>{c.chunk_id} · {c.words} words</span></div><p>{html.escape(c.text)}</p></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- Evaluation
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10), height=300, font=dict(color="#8E9AB2"))


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_DIR / "dataset.csv")


with tab_eval:
    df = load_dataset()
    st.markdown("#### Exploratory data analysis · dataset.csv")
    m = st.columns(5)
    m[0].metric("Retrieval records", f"{len(df)}")
    m[1].metric("Unique documents", f"{df.doc_id.nunique()}")
    m[2].metric("Mean retrieval score", f"{df.retrieval_score.mean():.3f}")
    m[3].metric("Mean answer quality", f"{df.answer_quality_score.mean():.3f}")
    m[4].metric("Answers ≥ 0.80", f"{(df.answer_quality_score >= 0.8).mean() * 100:.0f}%")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Retrieval score distribution**")
        fig = px.histogram(df, x="retrieval_score", nbins=10, color_discrete_sequence=[TEAL])
        fig.update_layout(**PLOT, bargap=0.08, yaxis_title="chunks")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Quality by query type**")
        g = df.groupby("query_type")[["answer_quality_score", "retrieval_score"]].mean().reset_index()
        fig = go.Figure([
            go.Bar(name="answer quality", x=g.query_type, y=g.answer_quality_score, marker_color=TEAL),
            go.Bar(name="retrieval score", x=g.query_type, y=g.retrieval_score, marker_color=SAFFRON),
        ])
        fig.update_layout(**PLOT, barmode="group", yaxis_range=[0.6, 0.9], legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Retrieval score vs answer quality**")
        fig = px.scatter(df, x="retrieval_score", y="answer_quality_score", color="query_type", opacity=0.75,
                         color_discrete_sequence=[TEAL, SAFFRON, SKY, ROSE, LIME])
        fig.update_layout(**PLOT, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with c4:
        st.markdown("**Chunks by document format**")
        g = df.groupby("doc_type").agg(chunks=("chunk_id", "count"), words=("word_count", "mean")).reset_index()
        fig = go.Figure([
            go.Bar(name="chunks", x=g.doc_type, y=g.chunks, marker_color=SKY),
            go.Scatter(name="mean word count", x=g.doc_type, y=g.words, yaxis="y2", mode="lines+markers", line=dict(color=SAFFRON, width=3)),
        ])
        fig.update_layout(**PLOT, yaxis2=dict(overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    r = df.retrieval_score.corr(df.answer_quality_score)
    best = df.groupby("query_type").answer_quality_score.mean().sort_values()
    st.markdown("**Key findings**")
    st.markdown(
        f"- The dataset is complete: {len(df)} rows, {df.shape[1]} columns, no missing values, spread over {df.doc_id.nunique()} documents and {df.doc_type.nunique()} formats.\n"
        f"- **{best.index[-1].title()}** queries get the best answers (mean {best.iloc[-1]:.3f}); **{best.index[0]}** queries score lowest ({best.iloc[0]:.3f}).\n"
        f"- Retrieval score and answer quality are uncorrelated (r = {r:.3f}): a high similarity score alone does not guarantee a good answer, so answers must cite their sources.\n"
        f"- Chunk length has almost no relation to answer quality (r = {df.word_count.corr(df.answer_quality_score):.3f})."
    )
    st.markdown("#### Live retriever benchmark")
    st.caption("16 labelled questions run against this app's own index. Hit@k = correct document in the top k results.")
    m = st.columns(4)
    m[0].metric("Hit@1", f"{b['hit1'] * 100:.0f}%")
    m[1].metric("Hit@3", f"{b['hit3'] * 100:.0f}%")
    m[2].metric("Mean reciprocal rank", f"{b['mrr']:.3f}")
    m[3].metric("Mean retrieval latency", f"{b['latency_ms']:.1f} ms")
    st.dataframe(pd.DataFrame(b["rows"]), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- Architecture
with tab_arch:
    st.markdown("#### From question to cited answer")
    nodes = [
        ("ing", "Ingest", "Load", "PDF, Word, TXT, HTML, Markdown parsed with pypdf / python-docx."),
        ("ing", "Ingest", "Chunk", "LangChain RecursiveCharacterTextSplitter, ~700 chars, 100 overlap."),
        ("ing", "Ingest", "Embed & index", "ChromaDB (all-MiniLM-L6-v2 vectors) + BM25 inverted index."),
        ("qry", "Query", "Retrieve", "Hybrid score = 0.6 × BM25 + 0.4 × cosine similarity, top-k."),
        ("qry", "Query", "Prompt", "Numbered chunks + rules: answer only from context, cite [n]."),
        ("qry", "Query", "Generate", "Groq / Grok / OpenAI / Ollama; extractive fallback offline."),
        ("qry", "Query", "Cite", "Answer shown with citations next to the source chunks."),
    ]
    cols = st.columns(7)
    for col, (cls, k, t, p) in zip(cols, nodes):
        col.markdown(f'<div class="node {cls}"><div class="k">{k}</div><b>{t}</b><p>{p}</p></div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Data flow diagram · level 0**")
        st.code(
            """  User ──question──▶ ┌───────────────────────┐
       ◀─answer+src── │  RAG Document Q&A     │ ──prompt+context──▶ LLM
  Uploader ──files──▶ │  System               │ ◀──generated text── (Groq/Grok)
                      └──────────┬────────────┘
                                 │
                      D1 Vector index (ChromaDB + BM25)""",
            language=None,
        )
        st.markdown("**Ethics & safety**")
        st.markdown(
            "- Every answer is shown with the exact source chunks it came from.\n"
            "- The prompt forbids outside knowledge; missing answers are reported as not found.\n"
            "- Not for legal or medical advice without expert review.\n"
            "- API keys stay in server secrets or the user's session; uploads are kept in memory only.\n"
            "- No personal data is stored (DPDP Act 2023)."
        )
    with a2:
        st.markdown("**Technology stack**")
        st.table(pd.DataFrame(
            [
                ("Frontend / UI", "Streamlit"),
                ("Backend / logic", "Python, LangChain text splitters"),
                ("Vector store", "ChromaDB + BM25 (rank-bm25)"),
                ("LLM", "Groq (Llama 3.3) / Grok (xAI) / OpenAI / Ollama via OpenAI-compatible API"),
                ("Parsing", "pypdf, python-docx"),
                ("Visualisation", "Plotly"),
                ("Dataset", "dataset.csv · 500 rows × 7 columns"),
            ],
            columns=["Layer", "Technology"],
        ).set_index("Layer"))
        st.markdown("**Prompt template**")
        st.code(build_prompt("{user question}", []).replace("CONTEXT:\n", "CONTEXT:\n[1..k] ({document title}, {chunk id})\n{chunk text}"), language=None)

st.caption("RAG-Based Document Q&A System · HCLTech Career Shaper Programme 2025-26 · Project No. 72")
