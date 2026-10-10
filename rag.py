"""Core RAG logic: YouTube transcript -> chunks -> FAISS -> Gemini answer."""
import os
import re
import time
from functools import lru_cache
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()  # must run before the os.getenv() settings below

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi

# Model names change over time; override in .env if one is retired.
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a YouTube video using ONLY the transcript "
            "excerpts below. If the answer is not in them, say you don't know. "
            "Be concise.\n\nTranscript excerpts:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def extract_video_id(url_or_id: str) -> str:
    """Accepts a full YouTube URL (watch, youtu.be, shorts, embed) or a bare 11-char ID."""
    s = url_or_id.strip()
    if re.fullmatch(r"[\w-]{11}", s):
        return s
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", s)
    if not m:
        raise ValueError("Could not find a YouTube video ID in that input.")
    return m.group(1)


def fetch_transcript(video_id: str, languages=("en", "hi", "te")) -> List[dict]:
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=list(languages))
    return [{"text": s.text, "start": s.start} for s in fetched]


def build_documents(snippets: List[dict], video_id: str,
                    chunk_chars: int = 1500, overlap_snippets: int = 2) -> List[Document]:
    """Group transcript lines into ~chunk_chars chunks, keeping the start time."""
    docs, buf, size, i = [], [], 0, 0
    while i < len(snippets):
        sn = snippets[i]
        buf.append(sn)
        size += len(sn["text"]) + 1
        i += 1
        if size >= chunk_chars or i == len(snippets):
            text = " ".join(x["text"].replace("\n", " ") for x in buf)
            start = int(buf[0]["start"])
            docs.append(Document(page_content=text,
                                 metadata={"start": start, "video_id": video_id}))
            # small overlap so answers spanning chunk edges aren't lost
            buf = buf[-overlap_snippets:] if overlap_snippets else []
            size = sum(len(x["text"]) + 1 for x in buf)
    return docs


def _is_rate_limit(e: Exception) -> bool:
    s = str(e)
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower()


def _with_retry(fn, tries: int = 6):
    """Call fn(); on a 429 wait (growing) and try again."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if not _is_rate_limit(e) or attempt == tries - 1:
                raise
            time.sleep(20 * (attempt + 1))


def build_vectorstore(docs: List[Document], api_key: str,
                      batch_size: int = 15, pause: float = 12.0) -> FAISS:
    """Embed in small batches with a pause so the free-tier per-minute quota isn't blown."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key)
    store = None
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        if store is None:
            store = _with_retry(lambda: FAISS.from_documents(batch, embeddings))
        else:
            _with_retry(lambda: store.add_documents(batch))
        if i + batch_size < len(docs):
            time.sleep(pause)
    return store


@lru_cache(maxsize=1)
def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=api_key, temperature=0.2)


def answer(question: str, vectorstore: FAISS, api_key: str, k: int = 4) -> Tuple[str, List[Document]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata['start']//60}:{d.metadata['start']%60:02d}] {d.page_content}"
                          for d in docs)
    chain = PROMPT | get_llm(api_key) | StrOutputParser()
    result = _with_retry(lambda: chain.invoke({"context": context, "question": question}))
    return result, docs