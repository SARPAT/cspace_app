"""PolicyPal — HR policy assistant built on the RAG-Based Document Q&A System (HCLTech Project 72)."""
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
    NOT_FOUND,
    PROVIDERS,
    Document,
    HybridIndex,
    build_prompt,
    chunk_text,
    extract_text,
    extractive_answer,
    list_models,
    load_builtin,
    run_benchmark,
    stream_answer,
)

st.set_page_config(page_title="PolicyPal · HR Policy Assistant", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- styles
INK, MUTED, LINE, BG = "#17161C", "#6B6876", "#E6E2D9", "#F6F4EF"
INDIGO, CORAL, GREEN, AMBER, SAND = "#3D3BBF", "#E5603B", "#1F8A5B", "#C08A1E", "#9C93C9"
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap');
:root {{ --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --bg:{BG}; --indigo:{INDIGO}; --coral:{CORAL}; }}
html, body, .stApp, .stMarkdown, button, input, textarea {{ font-family: 'Geist', system-ui, sans-serif; }}
.stApp {{ background: var(--bg); color: var(--ink); }}
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], [data-testid="stDecoration"] {{ display: none; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1080px; padding-top: 1.4rem; padding-bottom: 6rem; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid var(--line); }}
h1, h2, h3, h4 {{ color: var(--ink); letter-spacing: -.01em; }}
.serif {{ font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; }}
.mono {{ font-family: 'Geist Mono', ui-monospace, monospace; }}

/* top bar */
.topbar {{ display:flex; align-items:center; justify-content:space-between; padding: 4px 0 18px; border-bottom: 1px solid var(--line); margin-bottom: 8px; }}
.brand {{ display:flex; align-items:center; gap:10px; font: 600 17px 'Geist', sans-serif; color: var(--ink); }}
.mark {{ width:28px; height:28px; border-radius:8px; background: var(--ink); color:#fff; display:grid; place-items:center; font: italic 19px/1 'Instrument Serif', serif; }}
.brand small {{ font: 400 13px 'Geist', sans-serif; color: var(--muted); margin-left: 4px; }}
.status {{ display:flex; gap:8px; align-items:center; font: 500 12px 'Geist Mono', monospace; color: var(--muted); }}
.dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; background:{GREEN}; box-shadow: 0 0 0 3px rgba(31,138,91,.14); }}
.dot.off {{ background:{AMBER}; box-shadow: 0 0 0 3px rgba(192,138,30,.16); }}

/* tabs as a quiet segmented control */
.stTabs [role="tablist"] {{ gap: 2px; background: #ECE8DF; padding: 4px; border-radius: 12px; width: fit-content; border: 0; margin-top: 4px; }}
.stTabs [role="tab"] {{ padding: 7px 16px; border-radius: 9px; color: var(--muted); background: transparent; transition: all .15s ease; }}
.stTabs [role="tab"] p {{ font-weight: 500; font-size: 14px; }}
.stTabs [role="tab"]:hover {{ color: var(--ink); }}
.stTabs [role="tab"][aria-selected="true"] {{ background: #FFFFFF; color: var(--ink); box-shadow: 0 1px 2px rgba(23,22,28,.10); }}
.stTabs .react-aria-SelectionIndicator {{ display: none; }}
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before {{ display:none; }}

/* hero */
.hero {{ padding: 56px 0 18px; animation: rise .5s ease both; }}
.eyebrow {{ font: 500 11.5px 'Geist Mono', monospace; letter-spacing: .14em; text-transform: uppercase; color: var(--coral); }}
.hero h1 {{ font: 400 clamp(40px, 6vw, 68px)/1.02 'Instrument Serif', Georgia, serif; letter-spacing: -.02em; margin: 14px 0 14px; color: var(--ink); }}
.hero h1 em {{ color: var(--indigo); }}
.hero p {{ color: var(--muted); font-size: 16.5px; max-width: 520px; margin: 0; }}
@keyframes rise {{ from {{ opacity:0; transform: translateY(8px); }} to {{ opacity:1; transform:none; }} }}

/* pills */
[data-testid="stPills"] button, [data-testid="stButtonGroup"] button {{ border-radius: 999px !important; border: 1px solid var(--line) !important; background: #FFFFFF !important; color: var(--ink) !important; font-size: 13.5px !important; transition: all .15s ease; }}
[data-testid="stPills"] button:hover, [data-testid="stButtonGroup"] button:hover {{ border-color: var(--indigo) !important; color: var(--indigo) !important; }}

/* buttons */
.stButton button, [data-testid="stPopover"] button {{ border-radius: 10px; border: 1px solid var(--line); background: #FFFFFF; color: var(--ink); font-weight: 500; transition: all .15s ease; }}
.stButton button:hover, [data-testid="stPopover"] button:hover {{ border-color: var(--ink); color: var(--ink); }}

/* chat */
[data-testid="stChatMessage"] {{ background: transparent; padding: 6px 0; gap: 14px; animation: rise .35s ease both; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ flex-direction: row-reverse; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
  background: var(--ink); color: #F6F4EF; border-radius: 16px 16px 4px 16px; padding: 10px 16px; flex: 0 1 auto; max-width: 70%; margin-left: auto !important; margin-right: 0 !important; height: auto !important; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stVerticalBlock"] {{ height: auto !important; gap: 0; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdown"] > div, [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {{ color: #F6F4EF; margin: 0; }}
[data-testid="stChatMessageAvatarUser"] {{ display:none; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {{
  background: #FFFFFF; border: 1px solid var(--line); border-radius: 4px 16px 16px 16px; padding: 16px 20px 12px; }}
[data-testid="stChatMessageAvatarAssistant"] {{ background: var(--ink); color: #fff; border-radius: 8px; }}
[data-testid="stChatInput"] {{ border-radius: 14px; }}
[data-testid="stChatInput"] > div {{ border-radius: 14px; border-color: var(--line); background: #FFFFFF; box-shadow: 0 8px 30px rgba(23,22,28,.06); }}
[data-testid="stBottom"] > div {{ background: var(--bg); }}

.cite {{ font: 500 10.5px/1 'Geist Mono', monospace; color: var(--indigo); background: #ECEBFB; border-radius: 5px; padding: 2px 5px; margin: 0 1px; vertical-align: 2px; }}
.meta {{ display:flex; flex-wrap:wrap; gap: 14px; align-items:center; margin-top: 10px; padding-top: 10px; border-top: 1px solid #F0EDE6; font: 12px 'Geist Mono', monospace; color: var(--muted); }}
.meta .ok {{ color:{GREEN}; }} .meta .warn {{ color:{AMBER}; }}
.note {{ font-size: 12.5px; color: var(--muted); margin-top: 6px; }}

/* expander */
[data-testid="stExpander"] details {{ border: 1px solid var(--line); border-radius: 12px; background: #FFFFFF; }}
[data-testid="stDialog"] h2 {{ font: 400 34px/1.1 'Instrument Serif', Georgia, serif !important; }}
[data-testid="stExpander"] summary {{ font-size: 13.5px; font-weight: 500; }}

/* source cards */
.src {{ border-top: 1px solid #F0EDE6; padding: 12px 2px; }}
.src:first-child {{ border-top: 0; padding-top: 2px; }}
.src .h {{ display:flex; align-items:baseline; gap:10px; margin-bottom: 4px; }}
.src .n {{ font: 500 11px 'Geist Mono', monospace; color: var(--indigo); }}
.src .t {{ font-weight: 600; font-size: 14px; color: var(--ink); }}
.src .r {{ margin-left:auto; font: 11.5px 'Geist Mono', monospace; color: var(--muted); white-space: nowrap; }}
.src p {{ margin: 0; color: #3E3B47; font-size: 13.5px; line-height: 1.6; }}
.src mark {{ background: #FCEFC7; color: inherit; border-radius: 3px; padding: 0 2px; }}
.tag {{ font: 500 10px 'Geist Mono', monospace; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); border: 1px solid var(--line); border-radius: 5px; padding: 2px 6px; }}

/* cards (library, KPIs, steps) */
.card {{ background:#FFFFFF; border:1px solid var(--line); border-radius: 14px; padding: 18px 18px 14px; height: 100%; transition: border-color .15s ease, transform .15s ease; }}
.card:hover {{ border-color: #CFC9BC; transform: translateY(-1px); }}
.card .k {{ font: 500 10.5px 'Geist Mono', monospace; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }}
.card h3 {{ font: 400 23px/1.15 'Instrument Serif', Georgia, serif; margin: 8px 0 10px; padding: 0; }}
.card .f {{ font: 12px 'Geist Mono', monospace; color: var(--muted); }}
[class*="st-key-card-"] {{ background:#FFFFFF; border:1px solid var(--line); border-radius: 14px; padding: 18px 18px 8px; gap: 0; height: 100%; transition: border-color .15s ease, box-shadow .15s ease; }}
[class*="st-key-card-"]:hover {{ border-color: #CFC9BC; box-shadow: 0 6px 24px rgba(23,22,28,.06); }}
[class*="st-key-card-"] .k {{ font: 500 10.5px 'Geist Mono', monospace; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }}
[class*="st-key-card-"] h3.ct {{ font: 400 23px/1.15 'Instrument Serif', Georgia, serif; margin: 8px 0 8px; padding: 0; min-height: 54px; }}
[class*="st-key-card-"] .f {{ font: 12px 'Geist Mono', monospace; color: var(--muted); margin-bottom: 12px; }}
[class*="st-key-card-"] button {{ padding-left: 0; color: var(--indigo) !important; border: 0; background: transparent; }}
.kpi b {{ display:block; font: 400 40px/1 'Instrument Serif', Georgia, serif; color: var(--ink); margin-top: 10px; }}
.kpi span {{ font-size: 12.5px; color: var(--muted); }}
.step {{ border-left: 1px solid var(--line); padding: 2px 0 2px 14px; }}
.step .num {{ font: 500 11px 'Geist Mono', monospace; color: var(--coral); }}
.step b {{ display:block; font: 400 22px/1.2 'Instrument Serif', Georgia, serif; margin: 4px 0 4px; }}
.step p {{ font-size: 13px; color: var(--muted); margin: 0; }}
.section {{ font: 400 30px/1.1 'Instrument Serif', Georgia, serif; margin: 34px 0 4px; }}
.sub {{ color: var(--muted); font-size: 14px; margin: 0 0 16px; }}
.rows {{ border-top: 1px solid var(--line); }}
.rows div {{ display:flex; justify-content: space-between; gap: 16px; padding: 10px 0; border-bottom: 1px solid var(--line); font-size: 14px; }}
.rows div span:first-child {{ color: var(--muted); }}
.rows div span:last-child {{ text-align: right; }}
.foot {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--line); display:flex; flex-wrap:wrap; gap: 6px 18px; font: 12px 'Geist Mono', monospace; color: var(--muted); }}
@media (max-width: 700px) {{ .hero {{ padding-top: 28px; }} .brand small {{ display:none; }}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{ max-width: 88%; }} }}
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
    with st.spinner("Indexing the HR handbook…"):
        idx = HybridIndex()
        idx.add([Document(d.doc_id, d.title, d.category, d.doc_type, list(d.chunks)) for d in docs])
    st.session_state.index = idx
    st.session_state.history = []
    st.session_state.uploaded = set()
    st.session_state.bench = run_benchmark(idx, bench)
index: HybridIndex = st.session_state.index
UPLOADS = "Your uploads"


@st.cache_data(ttl=3600, show_spinner=False)
def available_models(provider: str, key: str) -> list[str]:
    try:
        return list_models(provider, key)
    except Exception:  # noqa: BLE001 - wrong key or no network: fall back to a typed model name
        return []


def add_files(files) -> None:
    for f in files or []:
        key = (f.name, f.size)
        if key in st.session_state.uploaded:
            continue
        text, dtype = extract_text(f.name, f.getvalue())
        chunks = chunk_text(text)
        st.session_state.uploaded.add(key)
        if not chunks:
            st.toast(f"No readable text in {f.name}. Scanned PDFs need OCR first.")
            continue
        n = sum(1 for d in index.docs if d.category == UPLOADS) + 1
        index.add([Document(f"UPL-{n:02d}", f.name.rsplit(".", 1)[0], UPLOADS, dtype, chunks)])
        st.toast(f"Added {f.name} · {len(chunks)} clauses")


# --------------------------------------------------------------------------- sidebar: settings
with st.sidebar:
    st.markdown("#### Settings")
    provider = st.selectbox("Answer engine", list(PROVIDERS), index=0)
    cfg = PROVIDERS[provider]
    secret_key = get_secret(cfg["secret"])
    if cfg["secret"] is None:
        api_key = ""
        st.caption("Uses Ollama on this computer: `ollama run llama3.2`.")
    elif secret_key:
        api_key = secret_key
        st.caption("API key loaded from app secrets.")
    else:
        api_key = st.text_input(
            f"{provider.split(' (')[0]} API key",
            type="password",
            help="Stays in this browser session only.",
            placeholder="Paste your key",
        )
        if provider.startswith("Groq"):
            st.caption("Free key at console.groq.com")
    models = available_models(provider, api_key) if (api_key or cfg["secret"] is None) else []
    if models:
        model = st.selectbox("Model", models, index=0)
    else:
        model = st.text_input("Model", value=cfg["model"])
    llm_ready = bool(api_key) or cfg["secret"] is None
    top_k = st.slider("Clauses per answer", 2, 8, 4)
    st.divider()
    st.caption("Search: BM25 + ChromaDB vectors" if index.dense_on else "Search: BM25 (vector model unavailable)")

engine_label = provider.split(" (")[0] if llm_ready else "Offline mode"

# --------------------------------------------------------------------------- helpers
STOPWORDS = {"the", "and", "for", "what", "who", "how", "are", "does", "with", "under", "this", "that", "from", "can", "get", "many", "much"}


def highlight(text: str, question: str) -> str:
    terms = {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", question)} - STOPWORDS
    out = html.escape(text)
    for t in sorted(terms, key=len, reverse=True):
        out = re.sub(rf"(?i)\b({re.escape(t)}\w*)", r"<mark>\1</mark>", out)
    return out


def cite_html(md: str) -> str:
    return re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', md)


def sources_html(hits, question) -> str:
    cards = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        right = "document opening" if h.header else f"{min(100, h.score * 100):.0f}% match"
        cards.append(
            f'<div class="src"><div class="h"><span class="n">[{i}]</span><span class="t">{html.escape(c.doc.title)}</span>'
            f'<span class="tag">{c.doc.doc_type}</span><span class="r">{right}</span></div>'
            f"<p>{highlight(c.text, question)}</p></div>"
        )
    return "".join(cards)


def render_turn(turn, latest: bool):
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        st.markdown(cite_html(turn["answer"]), unsafe_allow_html=True)
        missing = NOT_FOUND.lower()[:30] in turn["answer"].lower()
        state = '<span class="warn">● not in policy</span>' if missing else '<span class="ok">● grounded</span>'
        st.markdown(
            f'<div class="meta">{state}<span>{html.escape(turn["engine"])}</span>'
            f'<span>{turn["t_total"]:.2f} s</span><span>{len(turn["hits"])} sources</span></div>',
            unsafe_allow_html=True,
        )
        if turn.get("note"):
            st.markdown(f'<div class="note">{html.escape(turn["note"])}</div>', unsafe_allow_html=True)
        if turn["hits"]:
            with st.expander("Sources", expanded=latest):
                st.markdown(sources_html(turn["hits"], turn["q"]), unsafe_allow_html=True)


def generate(question: str, hits, turn) -> tuple[str, str]:
    if not llm_ready:
        turn["note"] = "Offline mode quotes the matching clauses. Add a Groq key in Settings for written answers."
        return extractive_answer(index, question, hits), "extractive · offline"
    placeholder = st.empty()
    placeholder.markdown('<span class="note">Reading the policies…</span>', unsafe_allow_html=True)
    prompt = build_prompt(question, hits)
    use_model = model
    try:
        pieces = stream_answer(provider, api_key, use_model, prompt)
        first = next(pieces, "")
    except Exception as e:  # noqa: BLE001 - retired model name: switch to a live one
        if type(e).__name__ not in ("NotFoundError", "BadRequestError"):
            raise
        alts = [m for m in available_models(provider, api_key) if m != model]
        if not alts:
            raise
        use_model = alts[0]
        turn["note"] = f"Model {model} is not available, so {use_model} was used."
        pieces = stream_answer(provider, api_key, use_model, prompt)
        first = next(pieces, "")
    text = first
    for piece in pieces:
        text += piece
        placeholder.markdown(cite_html(text) + " ▍", unsafe_allow_html=True)
    placeholder.empty()
    return text.strip(), f"{provider.split(' (')[0].lower()} · {use_model}"


def answer(question: str):
    t0 = time.perf_counter()
    base = index.retrieve(question, top_k)
    t_ret = (time.perf_counter() - t0) * 1000
    hits = index.with_headers(base) if base else []
    turn = {"q": question, "hits": hits, "t_retrieve": t_ret, "note": ""}
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        if not base:
            text, engine = NOT_FOUND, "no matching clauses"
        else:
            try:
                text, engine = generate(question, hits, turn)
            except Exception as e:  # noqa: BLE001 - show the reason, fall back to extractive
                text, engine = extractive_answer(index, question, hits), "extractive · fallback"
                turn["note"] = f"The model call failed ({type(e).__name__}: {str(e)[:140]}). Showing the offline answer."
    turn.update(answer=text, engine=engine, t_total=time.perf_counter() - t0)
    st.session_state.history.append(turn)
    st.rerun()


# --------------------------------------------------------------------------- top bar
policies = [d for d in index.docs if d.category != UPLOADS]
st.markdown(
    f"""<div class="topbar"><div class="brand"><span class="mark">P</span>PolicyPal<small>HR policy assistant</small></div>
<div class="status"><span class="dot {'' if llm_ready else 'off'}"></span>{html.escape(engine_label)}</div></div>""",
    unsafe_allow_html=True,
)

tab_ask, tab_lib, tab_ins, tab_how = st.tabs(["Ask", "Policies", "Insights", "How it works"])

# --------------------------------------------------------------------------- Ask
SAMPLES = [
    "How many days of earned leave do I get?",
    "What is the notice period if I resign?",
    "How do I raise a POSH complaint?",
    "What's the hotel limit on a work trip?",
    "Can I work remotely from Goa for a month?",
]
with tab_ask:
    hist = st.session_state.history
    if not hist:
        st.markdown(
            f"""<div class="hero"><div class="eyebrow">Orbitline HR handbook · {len(policies)} policies</div>
<h1>Ask HR anything.<br><em>Get the exact clause.</em></h1>
<p>Answers come only from company policy, with the source beside every line.</p></div>""",
            unsafe_allow_html=True,
        )
        pick = st.pills("Suggestions", SAMPLES, selection_mode="single", label_visibility="collapsed", key="pick0")
    else:
        pick = None
        lcol, rcol = st.columns([6, 1])
        with rcol:
            if st.button("New chat", width="stretch"):
                st.session_state.history = []
                st.rerun()
        for i, turn in enumerate(hist):
            render_turn(turn, latest=i == len(hist) - 1)

    c1, c2, _ = st.columns([1.7, 3, 3])
    with c1.popover("Add document", icon=":material/upload_file:", width="stretch"):
        st.caption("PDF, Word, TXT, Markdown or HTML. Kept in this session only.")
        add_files(
            st.file_uploader(
                "Upload", type=["pdf", "docx", "txt", "md", "html", "htm"], accept_multiple_files=True, label_visibility="collapsed"
            )
        )
    ups = [d for d in index.docs if d.category == UPLOADS]
    if ups:
        c2.markdown(
            f'<div class="status" style="height:40px">{len(ups)} uploaded · {html.escape(", ".join(d.title for d in ups))[:60]}</div>',
            unsafe_allow_html=True,
        )

    q = st.chat_input("Ask about leave, pay, travel, conduct…")
    if pick:
        answer(pick)
    elif q:
        answer(q)


# --------------------------------------------------------------------------- Policies
def open_policy(doc_id: str):
    d = next(d for d in index.docs if d.doc_id == doc_id)
    cs = [c for c in index.chunks if c.doc is d]

    def body():
        st.markdown(f'<div class="eyebrow">{html.escape(d.category)} · {d.doc_type} · {len(cs)} clauses</div>', unsafe_allow_html=True)
        st.markdown(
            "".join(f'<div class="src"><div class="h"><span class="n">§{c.idx + 1}</span><span class="r">{c.chunk_id}</span></div><p>{html.escape(c.text)}</p></div>' for c in cs),
            unsafe_allow_html=True,
        )

    st.dialog(d.title, width="large")(body)()


with tab_lib:
    st.markdown('<div class="section">The handbook</div>', unsafe_allow_html=True)
    words = sum(c.words for c in index.chunks)
    st.markdown(f'<p class="sub">{len(index.docs)} documents · {len(index.chunks)} clauses · {words:,} words indexed</p>', unsafe_allow_html=True)
    cats = list(dict.fromkeys(d.category for d in index.docs))
    pick_cat = st.segmented_control("Category", ["All"] + cats, default="All", label_visibility="collapsed")
    shown = [d for d in index.docs if pick_cat in (None, "All") or d.category == pick_cat]
    for row in range(0, len(shown), 3):
        cols = st.columns(3)
        for col, d in zip(cols, shown[row : row + 3]):
            n = sum(1 for c in index.chunks if c.doc is d)
            with col.container(key=f"card-{d.doc_id}"):
                st.markdown(
                    f'<div class="k">{html.escape(d.category)}</div><h3 class="ct">{html.escape(d.title)}</h3>'
                    f'<div class="f">{d.doc_type} · {n} clauses</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Read policy", key=f"read-{d.doc_id}", icon=":material/arrow_outward:", type="tertiary"):
                    open_policy(d.doc_id)
        st.write("")

# --------------------------------------------------------------------------- Insights
TIER_COLORS = {"Trusted": GREEN, "Usable": INDIGO, "Refer to HR": CORAL}
PLOT = dict(
    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=8, t=8, b=8), height=290, font=dict(family="Geist, sans-serif", color=MUTED, size=12),
)
AXES = dict(gridcolor="#ECE8DF", zerolinecolor="#ECE8DF", linecolor=LINE)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_DIR / "dataset_cleaned.csv")


def kpi(col, value: str, label: str):
    col.markdown(f'<div class="card kpi"><div class="k">{label}</div><b>{value}</b></div>', unsafe_allow_html=True)


def chart(fig):
    fig.update_layout(**PLOT)
    fig.update_xaxes(**AXES)
    fig.update_yaxes(**AXES)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


with tab_ins:
    df = load_dataset()
    st.markdown('<div class="section">How reliable are the answers?</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub">{len(df)} evaluated retrievals across {df.doc_id.nunique()} policy documents (dataset_cleaned.csv)</p>', unsafe_allow_html=True)
    k = st.columns(4)
    kpi(k[0], f"{df.retrieval_score.mean():.2f}", "Mean retrieval")
    kpi(k[1], f"{df.answer_quality_score.mean():.2f}", "Mean answer quality")
    kpi(k[2], f"{df.reliability_index.mean():.2f}", "Reliability index")
    kpi(k[3], f"{(df.confidence_tier == 'Trusted').mean() * 100:.0f}%", "Trusted answers")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confidence tiers**")
        t = df.confidence_tier.value_counts().reindex(list(TIER_COLORS)).reset_index()
        t.columns = ["tier", "n"]
        fig = go.Figure(go.Pie(labels=t.tier, values=t.n, hole=.68, marker=dict(colors=[TIER_COLORS[x] for x in t.tier], line=dict(color="#fff", width=2)), sort=False, textinfo="percent"))
        fig.update_layout(legend=dict(orientation="h", y=-0.08), showlegend=True)
        chart(fig)
    with c2:
        st.markdown("**Quality by policy area**")
        g = df.groupby("policy_area")[["retrieval_score", "answer_quality_score"]].mean().sort_values("answer_quality_score").reset_index()
        fig = go.Figure([
            go.Bar(name="retrieval", y=g.policy_area, x=g.retrieval_score, orientation="h", marker_color=SAND),
            go.Bar(name="answer quality", y=g.policy_area, x=g.answer_quality_score, orientation="h", marker_color=INDIGO),
        ])
        fig.update_layout(barmode="group", xaxis_range=[0.6, 0.9], legend=dict(orientation="h", y=-0.15), bargap=.3)
        chart(fig)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Retrieval vs answer quality**")
        fig = px.scatter(df, x="retrieval_score", y="answer_quality_score", color="confidence_tier", color_discrete_map=TIER_COLORS, opacity=.7,
                         category_orders={"confidence_tier": list(TIER_COLORS)},
                         labels={"retrieval_score": "retrieval score", "answer_quality_score": "answer quality", "confidence_tier": "tier"})
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(legend=dict(orientation="h", y=-0.2, title=None))
        chart(fig)
    with c4:
        st.markdown("**Answer quality by question type**")
        g = df.groupby("query_type").answer_quality_score.mean().sort_values().reset_index()
        fig = go.Figure(go.Bar(x=g.query_type, y=g.answer_quality_score, marker_color=[INDIGO] * (len(g) - 1) + [CORAL], text=g.answer_quality_score.round(3), textposition="outside"))
        fig.update_layout(yaxis_range=[0.7, 0.85])
        chart(fig)
    r = df.retrieval_score.corr(df.answer_quality_score)
    best = df.groupby("policy_area").answer_quality_score.mean().sort_values()
    st.markdown(
        f"- Only **{(df.confidence_tier == 'Trusted').mean() * 100:.0f}%** of answers are *Trusted*; **{(df.confidence_tier == 'Refer to HR').mean() * 100:.0f}%** should be checked with HR.\n"
        f"- Retrieval score barely predicts answer quality (r = {r:.2f}), which is why every answer shows its source.\n"
        f"- **{best.index[-1]}** policies get the best answers; **{best.index[0]}** the weakest."
    )

    b = st.session_state.bench
    st.markdown('<div class="section">Live retriever benchmark</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub">{len(b["rows"])} labelled HR questions run against this app\'s own index</p>', unsafe_allow_html=True)
    k = st.columns(4)
    kpi(k[0], f"{b['hit1'] * 100:.0f}%", "Hit@1")
    kpi(k[1], f"{b['hit3'] * 100:.0f}%", "Hit@3")
    kpi(k[2], f"{b['mrr']:.2f}", "MRR")
    kpi(k[3], f"{b['latency_ms']:.0f} ms", "Retrieval latency")
    st.write("")
    with st.expander("Benchmark questions"):
        st.dataframe(pd.DataFrame(b["rows"]), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- How it works
with tab_how:
    st.markdown('<div class="section">From question to cited clause</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Retrieval-augmented generation, in six steps</p>', unsafe_allow_html=True)
    steps = [
        ("Load", "PDF, Word, TXT, HTML and Markdown parsed."),
        ("Chunk", "700-char clauses, 100 overlap."),
        ("Index", "BM25 + ChromaDB MiniLM vectors."),
        ("Retrieve", "0.6 × BM25 + 0.4 × cosine, top-k."),
        ("Generate", "Groq Llama answers from clauses only."),
        ("Cite", "Every claim links to its source."),
    ]
    cols = st.columns(6)
    for i, (col, (t, p)) in enumerate(zip(cols, steps), 1):
        col.markdown(f'<div class="step"><div class="num">0{i}</div><b>{t}</b><p>{p}</p></div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown('<div class="section" style="font-size:24px">Stack</div>', unsafe_allow_html=True)
        stack = [
            ("Interface", "Streamlit"),
            ("Chunking", "LangChain text splitter"),
            ("Vector store", "ChromaDB + BM25"),
            ("Embeddings", "all-MiniLM-L6-v2"),
            ("LLM", "Groq · Llama 3.3 70B"),
            ("Parsing", "pypdf, python-docx"),
            ("Charts", "Plotly"),
        ]
        st.markdown('<div class="rows">' + "".join(f"<div><span>{a}</span><span>{b_}</span></div>" for a, b_ in stack) + "</div>", unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="section" style="font-size:24px">Guardrails</div>', unsafe_allow_html=True)
        guards = [
            ("Grounding", "Answers use retrieved clauses only"),
            ("Transparency", "Sources shown beside every answer"),
            ("Honesty", "Says so when policy is silent"),
            ("Privacy", "Uploads stay in memory, keys in secrets"),
            ("Scope", "Not legal advice; HR has the final word"),
        ]
        st.markdown('<div class="rows">' + "".join(f"<div><span>{a}</span><span>{b_}</span></div>" for a, b_ in guards) + "</div>", unsafe_allow_html=True)
    st.write("")
    with st.expander("Prompt template"):
        st.code(build_prompt("{employee question}", []).replace("POLICY EXTRACTS:\n", "POLICY EXTRACTS:\n[1..k] ({policy title}, {chunk id})\n{clause text}"), language=None)

st.markdown(
    """<div class="foot"><span>PolicyPal · RAG-Based Document Q&amp;A System · HCLTech Project 72</span>
<span>Saransh Patel · 2300320120225</span><span>Mentor: Sir S.P Gautam</span><span>ABES Engineering College</span></div>""",
    unsafe_allow_html=True,
)
