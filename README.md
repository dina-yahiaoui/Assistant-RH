$env:OPENAI_API_KEY="sk-or-v1-TA-CLE-OPENROUTER"
$env:OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
$env:OPENROUTER_EMBEDDINGS_MODEL="openai/text-embedding-3-small"
$env:QDRANT_URL="http://localhost:6333"
$env:QDRANT_COLLECTION="assistant_rh_cvs"
streamlit run app.py
