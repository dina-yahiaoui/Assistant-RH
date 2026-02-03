# app/ingest.py
from pathlib import Path
import pandas as pd

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

CV_CSV_PATH = BASE_DIR / "data" / "cv" / "resumes.csv"
PERSIST_DIR = BASE_DIR / "data" / "chroma_cv"


def load_cv_documents(limit: int = 50):
    """
    Charge les CV depuis le CSV Kaggle et les transforme
    en objets Document pour LangChain.
    Le code essaie de deviner automatiquement la bonne colonne texte.
    """
    print(f"Lecture du fichier CSV : {CV_CSV_PATH}")
    df = pd.read_csv(CV_CSV_PATH)
    df = df.head(limit)

    print("Colonnes trouvées dans le CSV :", list(df.columns))

    # 👉 On essaie de deviner la bonne colonne texte
    candidate_text_cols = [
        "Resume", "resume", "RESUME",
        "Resume_str", "ResumeText", "Resume_text",
        "CV", "cv",
        "Text", "text",
        "Summary", "summary",
        "Professional Summary",
    ]

    text_col = None
    for col in candidate_text_cols:
        if col in df.columns:
            text_col = col
            break

    # Si aucune colonne connue trouvée, on prend la dernière
    if text_col is None:
        text_col = df.columns[-1]

    print("Colonne utilisée pour le texte du CV :", text_col)

    docs = []
    for idx, row in df.iterrows():
        text = str(row[text_col])

        metadata = {
            "row_index": int(idx),
        }

        # On essaie aussi de récupérer une catégorie si elle existe
        for cat_col in ["Category", "category", "Job Title", "CategoryName"]:
            if cat_col in df.columns:
                metadata["category"] = str(row[cat_col])
                break

        docs.append(Document(page_content=text, metadata=metadata))

    print(f"{len(docs)} documents créés.")
    return docs


def build_vector_store():
    """
    Construit et persiste le vector store Chroma
    à partir des CV chargés.
    """
    docs = load_cv_documents()

    print("Création des embeddings (modèle local) et du vector store Chroma...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    vectordb.persist()
    print(f"Vector store créé et sauvegardé dans : {PERSIST_DIR}")
    return vectordb


if __name__ == "__main__":
    build_vector_store()
    print("✅ Indexation des CV terminée.")
