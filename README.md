# YouTube RAG Chat

Ask questions about any YouTube video. Pipeline: transcript -> timestamped chunks -> Gemini
embeddings -> FAISS -> Gemini (via LangChain) -> Streamlit chat.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your Gemini API key in .env
streamlit run app.py
```

Get a free API key at https://aistudio.google.com/apikey

## Notes
- Free tier has rate limits. If you hit a 429 error, wait a minute and retry.
- If a model name is retired, set `GEMINI_MODEL` / `GEMINI_EMBED_MODEL` in `.env`.
- Videos need captions available. Languages tried: en, hi, te (edit `fetch_transcript`).
- Answers include clickable timestamp sources.
