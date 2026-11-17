"""Student Support AI Chatbot (Streamlit front end).

Run:  streamlit run chatbot/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from chatbot.service import BackendUnavailable, ask, check_backend, make_pipeline  # noqa: E402
from common.config import DOCS_DIR, LLM_MODELS, PROMPTS_DIR  # noqa: E402
from llmops.monitoring import summary  # noqa: E402
from models.rag.ingest import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP, build_index, collection_name  # noqa: E402
from common.config import VECTOR_DB_DIR  # noqa: E402

st.set_page_config(page_title="Student Support Assistant", page_icon=":material/school:", layout="centered")

st.markdown("""
<style>
  .block-container {max-width: 820px; padding-top: 2rem;}
  .src {font-size: 0.82rem; color: #5b6170; border-left: 2px solid #d0d4dc; padding: 2px 0 2px 10px; margin: 6px 0;}
  .src b {color: #2b2f38;}
  .meta {font-size: 0.75rem; color: #8a909c;}
</style>""", unsafe_allow_html=True)

EXAMPLES = ["What is the minimum attendance to write the end semester exam?",
            "How many library books can an undergraduate borrow?",
            "What are the eligibility criteria for the Merit Scholarship?",
            "What is the dress code for students?"]


@st.cache_resource(show_spinner="Loading knowledge base and embedding model...")
def get_pipeline(llm, prompt_version, top_k, guardrail, index_version):
    return make_pipeline(llm=llm, prompt_version=prompt_version, chunk_words=DEFAULT_CHUNK_WORDS,
                         overlap=DEFAULT_OVERLAP, top_k=top_k, max_distance=0.65 if guardrail else None)


def index_exists():
    return (VECTOR_DB_DIR / f"{collection_name(DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP)}_manifest.json").exists()


# ---------------- sidebar: settings, documents, monitoring ----------------
with st.sidebar:
    st.subheader("Settings")
    llm = st.selectbox("Language model", LLM_MODELS, help="Runs locally through Ollama on CPU.")
    prompts = sorted(p.stem for p in PROMPTS_DIR.glob("rag_v*.yaml"))
    prompt_version = st.selectbox("Prompt version", prompts, index=len(prompts) - 1)
    top_k = st.slider("Passages retrieved", 1, 8, 4)
    guardrail = st.toggle("Relevance guardrail", value=False,
                          help="Refuse without calling the model when no passage is similar enough.")

    st.subheader("Knowledge base")
    docs = sorted(p.name for p in DOCS_DIR.iterdir() if p.suffix.lower() in {".pdf", ".txt", ".md"})
    st.caption("\n".join(f"- {d}" for d in docs) or "No documents yet.")
    uploads = st.file_uploader("Add university documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Rebuild index", icon=":material/refresh:", use_container_width=True, disabled=not (uploads or not index_exists())):
        with st.spinner("Extracting, chunking and embedding documents..."):
            try:
                for f in uploads or []:
                    (DOCS_DIR / Path(f.name).name).write_bytes(f.getbuffer())
                stats = build_index(DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP)
                st.session_state.index_version = st.session_state.get("index_version", 0) + 1
                st.success(f"Indexed {len(stats['files'])} file(s), {stats['chunks']} chunks.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.subheader("Monitoring")
    s = summary()
    c1, c2 = st.columns(2)
    c1.metric("Requests", s.get("requests", 0))
    c2.metric("Not found", f"{s.get('refusal_rate', 0):.0%}" if s.get("requests") else "-")
    c1.metric("p50 latency", f"{s['latency_p50_s']}s" if s.get("latency_p50_s") is not None else "-")
    c2.metric("Errors", f"{s.get('error_rate', 0):.0%}" if s.get("requests") else "-")
    if st.button("Clear conversation", icon=":material/delete_sweep:", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- main: chat ----------------
st.title("Student Support Assistant")
st.caption("Answers come only from the CHRIST University handbook and documents you add. Every answer cites its pages.")

if not index_exists():
    st.warning("The knowledge base has not been built yet. Use **Rebuild index** in the sidebar.")
    st.stop()

try:
    check_backend(llm)
except BackendUnavailable as e:
    st.error(str(e))
    st.stop()

pipe = get_pipeline(llm, prompt_version, top_k, guardrail, st.session_state.get("index_version", 0))
st.session_state.setdefault("messages", [])


def render_answer(m):
    if m.get("not_found"):
        st.warning(m["content"], icon=":material/search_off:")
    else:
        st.markdown(m["content"])
    if m.get("sources"):
        with st.expander(f"Sources ({len(m['sources'])} passages)"):
            for h in m["sources"]:
                snippet = h["text"][:260] + ("..." if len(h["text"]) > 260 else "")
                st.markdown(f"<div class='src'><b>{h['source']}, p. {h['page']}</b> &middot; similarity "
                            f"{1 - h['distance']:.2f}<br>{snippet}</div>", unsafe_allow_html=True)
    if m.get("meta"):
        st.markdown(f"<div class='meta'>{m['meta']}</div>", unsafe_allow_html=True)


for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/school:"):
        render_answer(m) if m["role"] == "assistant" else st.markdown(m["content"])

if not st.session_state.messages:
    st.markdown("**Try asking**")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

question = st.chat_input("Ask about attendance, exams, library, scholarships...") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/school:"):
        try:
            with st.spinner("Searching the handbook and writing an answer..."):
                r = ask(pipe, question)
            msg = {"role": "assistant", "content": r["text"], "not_found": r["not_found"], "sources": r["sources"],
                   "meta": f"{r['llm']} &middot; {r['prompt_version']} &middot; {r['latency_s']}s"}
        except Exception as e:
            msg = {"role": "assistant", "not_found": True,
                   "content": f"Something went wrong while answering: {type(e).__name__}: {e}"}
        render_answer(msg)
    st.session_state.messages.append(msg)
