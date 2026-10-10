import os

import streamlit as st

import rag  # loads .env on import

API_KEY = os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(page_title="YouTube RAG Chat", page_icon="🎬", layout="wide")
st.title("🎬 Chat with a YouTube video")

if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Add it to a `.env` file (see `.env.example`) and restart.")
    st.stop()

with st.sidebar:
    st.header("Video")
    url = st.text_input("YouTube URL or video ID")
    k = st.slider("Chunks retrieved", 2, 8, 4)
    if st.button("Load transcript", type="primary", disabled=not url):
        try:
            with st.spinner("Fetching transcript and building index..."):
                vid = rag.extract_video_id(url)
                snippets = rag.fetch_transcript(vid)
                docs = rag.build_documents(snippets, vid)
                st.session_state.vs = rag.build_vectorstore(docs, API_KEY)
                st.session_state.vid = vid
                st.session_state.messages = []
            st.success(f"Indexed {len(docs)} chunks.")
        except Exception as e:
            st.error(f"Failed: {e}")
    if "vid" in st.session_state:
        st.video(f"https://www.youtube.com/watch?v={st.session_state.vid}")
    st.caption(f"LLM: `{rag.LLM_MODEL}`")

if "vs" not in st.session_state:
    st.info("Paste a YouTube link in the sidebar and click **Load transcript**.")
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                st.markdown(m["sources"])

if q := st.chat_input("Ask something about the video"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                ans, docs = rag.answer(q, st.session_state.vs, API_KEY, k=k)
            vid = st.session_state.vid
            src = "\n\n".join(
                f"- [{d.metadata['start']//60}:{d.metadata['start']%60:02d}]"
                f"(https://www.youtube.com/watch?v={vid}&t={d.metadata['start']}s): "
                f"{d.page_content[:200]}..."
                for d in docs
            )
            st.markdown(ans)
            with st.expander("Sources"):
                st.markdown(src)
            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": src})
        except Exception as e:
            st.error(f"LLM call failed: {e}")
