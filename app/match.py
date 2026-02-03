# app/match.py
from pathlib import Path
from typing import List, Dict

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from openai import OpenAI
import os

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "data" / "chroma_cv"


def load_vectordb() -> Chroma:
    """
    Charge le vector store Chroma déjà construit par ingest.py.
    """
    print(f"Chargement du vector store depuis : {PERSIST_DIR}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    return vectordb


def retrieve_candidates(job_offer: str, k: int = 5) -> List[Document]:
    """
    Récupère les k CV les plus proches de l'offre.
    """
    vectordb = load_vectordb()
    docs = vectordb.similarity_search(job_offer, k=k)
    print(f"{len(docs)} candidats récupérés depuis Chroma.")
    return docs


def get_openrouter_client() -> OpenAI:
    """
    Crée un client OpenRouter avec la clé stockée dans OPENAI_API_KEY.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "La variable d'environnement OPENAI_API_KEY n'est pas définie. "
            "Mets ta clé OpenRouter avec : $env:OPENAI_API_KEY=\"sk-or-v1-...\""
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client


def analyze_with_llm(job_offer: str, docs: List[Document]) -> List[Dict]:
    """
    Appelle directement OpenRouter (GPT-4o mini) pour analyser chaque CV.
    """
    client = get_openrouter_client()

    results = []
    for doc in docs:
        prompt = f"""
Tu es un assistant RH pour l'alternance.

Voici l'offre d'alternance :

\"\"\"{job_offer}\"\"\"

Voici le CV d'un candidat :

\"\"\"{doc.page_content[:2000]}\"\"\"

Analyse ce candidat par rapport à l'offre et réponds en FRANÇAIS
avec le format SUIVANT (texte simple, pas de JSON) :

Score (0-100) : <un nombre>
Points forts :
- ...
- ...
Points faibles :
- ...
- ...

Ne rajoute rien d'autre autour, seulement ce format.
"""

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert RH."},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content

        results.append({
            "metadata": doc.metadata,
            "analysis": content,
        })

    return results


def match(job_offer: str, k: int = 3) -> List[Dict]:
    """
    Pipeline complet :
    1) retrieval des CV les plus proches
    2) analyse RH par le LLM (OpenRouter)
    """
    docs = retrieve_candidates(job_offer, k=k)
    analyses = analyze_with_llm(job_offer, docs)
    return analyses


if __name__ == "__main__":
    # Petit test en ligne de commande
    default_offer = """
Nous recherchons un alternant Data / IA maîtrisant Python, SQL,
et ayant déjà réalisé des projets de data analysis ou machine learning.
"""

    print("=== TEST MATCHING EN LIGNE DE COMMANDE AVEC OPENROUTER ===")
    print("Offre utilisée :")
    print(default_offer)
    print("==========================================\n")

    results = match(default_offer, k=3)

    for i, res in enumerate(results, start=1):
        print(f"--- Candidat #{i} ---")
        print("Métadonnées :", res["metadata"])
        print("Analyse :")
        print(res["analysis"])
        print("------------------------------\n")
