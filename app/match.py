# app/match.py
import os
from pathlib import Path
from typing import List, Dict

from qdrant_client import QdrantClient

from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# --- Chemins ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Config Qdrant & OpenRouter ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "assistant_rh_cvs")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_EMBEDDINGS_MODEL = os.getenv(
    "OPENROUTER_EMBEDDINGS_MODEL",
    "openai/text-embedding-3-small",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_qdrant_vectordb() -> QdrantVectorStore:
    """
    Charge la collection Qdrant existante avec les embeddings OpenRouter.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY n'est pas défini. "
            "Mets ta clé OpenRouter dans OPENAI_API_KEY (sk-or-v1-...)."
        )

    embeddings = OpenAIEmbeddings(
        model=OPENROUTER_EMBEDDINGS_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    # ✅ Création explicite du client Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
    )

    # ✅ Passage du client au VectorStore
    vectordb = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )

    return vectordb



def retrieve_candidates(job_offer: str, k: int = 3):
    """
    Récupère des chunks depuis Qdrant et les regroupe par candidat.
    Retourne k candidats uniques avec leurs chunks concaténés.
    """
    vectordb = load_qdrant_vectordb()

    # On récupère large
    raw_docs = vectordb.similarity_search(job_offer, k=50)

    candidates = {}
    for doc in raw_docs:
        cid = doc.metadata.get("candidate_index")

        if cid not in candidates:
            candidates[cid] = {
                "metadata": doc.metadata,
                "texts": []
            }

        candidates[cid]["texts"].append(doc.page_content)

        if len(candidates) >= k:
            break

    print(f"{len(candidates)} candidats uniques sélectionnés.")
    return candidates



def get_openrouter_client() -> OpenAI:
    api_key = OPENAI_API_KEY
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY n'est pas défini pour le LLM OpenRouter."
        )
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )
    return client


def analyze_with_llm(job_offer: str, candidates_dict) -> list:
    client = get_openrouter_client()
    results = []

    for cid, data in candidates_dict.items():
        full_text = "\n".join(data["texts"])[:4000]  # limite tokens
        meta = data["metadata"]

        prompt = f"""
Tu es un assistant RH.

Voici l'offre :

\"\"\"{job_offer}\"\"\"

Voici le CV complet du candidat (plusieurs extraits regroupés) :

\"\"\"{full_text}\"\"\"

Analyse ce candidat et réponds STRICTEMENT avec :

Score (0-100) : <nombre>
Points forts :
- ...
Points faibles :
- ...
"""

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert RH."},
                {"role": "user", "content": prompt},
            ],
        )

        results.append({
            "metadata": meta,
            "analysis": response.choices[0].message.content
        })

    return results



def match(job_offer: str, k: int = 3) -> List[Dict]:
    """
    Pipeline complet :
    1) retrieval des chunks/candidats les plus proches dans Qdrant
    2) analyse RH par le LLM (OpenRouter)
    """
    candidates = retrieve_candidates(job_offer, k)
    return analyze_with_llm(job_offer, candidates)


if __name__ == "__main__":
    default_offer = """
Nous recherchons un alternant Data / IA maîtrisant Python, SQL,
et ayant déjà réalisé des projets de data analysis ou machine learning.
"""

    print("=== TEST MATCHING AVEC QDRANT + OPENROUTER ===")
    print(default_offer)
    print("==============================================\n")

    results = match(default_offer, k=3)

    for i, res in enumerate(results, start=1):
        print(f"--- Candidat/Chunk #{i} ---")
        print("Métadonnées :", res["metadata"])
        print("Analyse :")
        print(res["analysis"])
        print("------------------------------\n")
