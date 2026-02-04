# app/ingest.py
import os
from pathlib import Path
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# --- Chemins ---
BASE_DIR = Path(__file__).resolve().parent.parent
CV_CSV_PATH = BASE_DIR / "data" / "cv" / "resumes.csv"

# --- Config Qdrant & OpenRouter ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "assistant_rh_cvs")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_EMBEDDINGS_MODEL = os.getenv(
    "OPENROUTER_EMBEDDINGS_MODEL",
    "openai/text-embedding-3-small",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def detect_text_column(df: pd.DataFrame) -> str:
    """
    Essaie de deviner la colonne qui contient le texte du CV.
    """
    candidate_text_cols = [
        "Resume", "resume", "RESUME",
        "Resume_str", "ResumeText", "Resume_text",
        "CV", "cv",
        "Text", "text",
        "Summary", "summary",
        "Professional Summary",
    ]
    for col in candidate_text_cols:
        if col in df.columns:
            return col
    # fallback : dernière colonne
    return df.columns[-1]


def load_cv_documents(limit: int = 10):
    """
    Charge jusqu'à 10 profils Kaggle (CSV) et les transforme en Documents.
    On essaie de mélanger les catégories si possible.
    """
    print(f"Lecture du CSV Kaggle : {CV_CSV_PATH}")
    df = pd.read_csv(CV_CSV_PATH)
    print("Colonnes CSV :", list(df.columns))

    text_col = detect_text_column(df)
    print("Colonne utilisée pour le texte du CV :", text_col)

    # Si la colonne Category existe, on essaie de prendre 1 CV par catégorie
    if "Category" in df.columns:
        grouped = (
            df.groupby("Category", group_keys=False)
              .apply(lambda x: x.sample(1, random_state=42))
        )
        df_sample = grouped.head(limit)
    else:
        # Sinon, on prend juste 10 CV au hasard
        df_sample = df.sample(limit, random_state=42)

    print("Catégories retenues :", df_sample.get("Category", "N/A").tolist())

    docs = []
    for idx, row in df_sample.iterrows():
        text = str(row[text_col])
        metadata = {
            "candidate_index": int(idx),
        }

        if "Category" in df.columns:
            metadata["category"] = str(row["Category"])

        docs.append(Document(page_content=text, metadata=metadata))

    print(f"{len(docs)} CV chargés (avant découpage).")
    return docs



def build_qdrant_vector_store():
    """
    Construit / recrée la collection Qdrant à partir de 10 profils Kaggle.
    Avec découpage en chunks (RAG-style).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY n'est pas défini. "
            "Mets ta clé OpenRouter dans OPENAI_API_KEY (sk-or-v1-...)."
        )

    base_docs = load_cv_documents(limit=10)

    # --- Découpage en chunks ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(base_docs)
    print(f"{len(chunks)} chunks générés à partir des 10 CV.")

    # Optionnel : afficher un exemple
    example = chunks[0]
    print("Exemple de chunk :")
    print("Metadata :", example.metadata)
    print("Texte (début) :", example.page_content[:200], "...\n")

    print("Initialisation des embeddings OpenRouter pour Qdrant...")
    embeddings = OpenAIEmbeddings(
        model=OPENROUTER_EMBEDDINGS_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    print(f"Indexation de {len(chunks)} chunks dans Qdrant ({QDRANT_URL}, collection '{QDRANT_COLLECTION}')...")

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        prefer_grpc=False,  # HTTP suffit
        collection_name=QDRANT_COLLECTION,
        force_recreate=True,  # recrée la collection à chaque run
    )

    print("✅ Indexation terminée dans Qdrant.")
    return vector_store


if __name__ == "__main__":
    build_qdrant_vector_store()
