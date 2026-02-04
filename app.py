# app.py
import streamlit as st

from app.ingest import build_qdrant_vector_store
from app.match import match



st.set_page_config(page_title="Assistant RH Alternance", layout="wide")

st.title("🤖 Assistant RH – Matching d'alternants avec RAG + LLM")

st.markdown(
    """
Cet assistant permet à un service RH de :

1. **Indexer** une base de CV candidats (dataset Kaggle) dans une base vectorielle Qdrant
2. **Analyser** une offre d'alternance  
3. **Retrouver** les profils les plus pertinents  
4. **Générer** pour chaque candidat une analyse via un **LLM (OpenRouter, gpt-4o-mini)** :
   - Score (0-100)
   - Points forts
   - Points faibles
"""
)

st.sidebar.header("Actions")

if st.sidebar.button("📥 (Ré)indexer les CV dans Qdrant"):
    with st.spinner("Indexation des CV en cours..."):
        build_qdrant_vector_store()
    st.sidebar.success("Indexation terminée ✅")


st.markdown("### 1. Coller une offre d'alternance")

default_offer = """Nous recherchons un alternant Data / IA maîtrisant Python, SQL,
et ayant déjà réalisé des projets de data analysis ou machine learning."""

job_offer = st.text_area(
    "Offre d'alternance",
    value=default_offer,
    height=200,
    help="Colle ici une offre d'alternance (par exemple copiée depuis HelloWork).",
)

st.markdown("### 2. Choisir le nombre de profils à proposer")

k = st.slider(
    "Nombre de candidats à retourner",
    min_value=1,
    max_value=10,
    value=3,
)

st.markdown("### 3. Lancer le matching")

if st.button("🔍 Lancer le matching"):
    if not job_offer.strip():
        st.warning("Merci de coller une offre d'alternance.")
    else:
        with st.spinner("Analyse en cours avec le RAG + LLM..."):
            results = match(job_offer, k=k)

        st.subheader("Résultats du matching")

        if not results:
            st.info("Aucun candidat trouvé.")
        else:
            for i, res in enumerate(results, start=1):
                meta = res["metadata"]
                analysis = res["analysis"]

                st.markdown(
                    f"#### 🧑‍🎓 Candidat #{i} — "
                    f"index: `{meta.get('candidate_index')}` | "
                    f"catégorie: `{meta.get('category', 'N/A')}`"
                )

                st.markdown("**Analyse générée par le LLM :**")
                st.code(analysis)
                st.markdown("---")
