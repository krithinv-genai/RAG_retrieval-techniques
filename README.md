# Grounded RAG Retrieval Comparison

A small Streamlit demo that compares four retrieval strategies on local or uploaded text:

- Dense search with FAISS
- Sparse search with BM25
- Maximal Marginal Relevance (MMR)
- Hybrid dense/sparse search with cross-encoder re-ranking

The app shows retrieved chunks, scores, metadata, and grounded answers with citation checks. If Groq credentials are not configured, it uses a local extractive fallback.

## Requirements

- Python 3.12+
- Internet access for Hugging Face models and optional Groq API access

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set the Hugging Face embedding and cross-encoder model names in `.env`. Add `HF_TOKEN` for gated Hugging Face models if needed.

Groq is optional. Add `GROQ_API_KEY` and `GROQ_MODEL_NAME` to enable generated answers; otherwise the app uses the local fallback.

## Run

```powershell
streamlit run app.py
```

Choose the sample document or upload up to three `.txt` files, enter a question, and select **Compare Retrievers**. Retrieval and model settings can be adjusted in the sidebar.
