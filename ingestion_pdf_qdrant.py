import os
import dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

dotenv.load_dotenv()

# --- Config OpenRouter ---
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_EMBEDDINGS_MODEL = os.getenv("OPENROUTER_EMBEDDINGS_MODEL")

# --- Config Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "cv_chunks"

print(f"OPENROUTER_BASE_URL    : {OPENROUTER_BASE_URL}")
print(f"OPENROUTER_API_KEY     : {OPENROUTER_API_KEY[:10]}...")
print(f"OPENROUTER_EMBEDDINGS  : {OPENROUTER_EMBEDDINGS_MODEL}")
print(f"QDRANT_URL             : {QDRANT_URL}")

# --- Initialisation embeddings ---
embeddings = OpenAIEmbeddings(
    model=OPENROUTER_EMBEDDINGS_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

# --- Chargement du PDF ---
loader = PDFPlumberLoader(file_path="./cvs/11580408.pdf")
docs = loader.load()

print(f"\nTotal caractères : {len(docs[0].page_content)}")

# --- Découpage en chunks ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
chunks = text_splitter.split_documents(docs)

print(f"Split en {len(chunks)} sous-documents.")

# --- Ajout de métadonnées ---
DUMMY_SKILLS = ["python", "java", "react", "sql", "aws"]

for chunk in chunks:
    chunk.metadata["skills"] = DUMMY_SKILLS

# --- Affichage des chunks avant indexation ---
for i, chunk in enumerate(chunks):
    print("*" * 80)
    print(f"Chunk {i}: longueur {len(chunk.page_content)}")
    print(f"Metadata: {chunk.metadata}")
    print("*" * 80)
    print(f"{chunk.page_content}\n")

# --- Indexation dans Qdrant ---
# ⚠️ force_recreate=True : recréer la collection à chaque exécution (pratique en dev).
# En production, retirer cette option pour préserver les données existantes.
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    force_recreate=True,
)

# --- Vérification : afficher le contenu indexé via le client natif ---
client = QdrantClient(url=QDRANT_URL)

collection_info = client.get_collection(COLLECTION_NAME)

print("\n" + "=" * 100)
print(f"  QDRANT — Collection '{COLLECTION_NAME}' : {collection_info.points_count} point(s) indexé(s)")
print("=" * 100)

# Scroll pour récupérer tous les points avec payload + vecteurs
points, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=100,
    with_payload=True,
    with_vectors=True,
)

for point in points:
    payload = point.payload or {}

    # langchain_qdrant stocke le texte dans "page_content"
    # et les métadonnées (dont "skills") dans un sous-dict "metadata"
    text     = payload.get("page_content", "")
    metadata = payload.get("metadata", {})
    vector   = point.vector or []

    print(f"\n{'─' * 100}")
    print(f"  ID              : {point.id}")
    print(f"  Metadata        : {metadata}")
    print(f"  Skills (liste)  : {metadata.get('skills', [])}")
    print(f"  Vector dim      : {len(vector)}")
    print(f"  Vector (x8)     : [{', '.join(f'{v:+.4f}' for v in vector[:8])}…]")
    print(f"  Content snippet : {text[:120]}…")
    print(f"{'─' * 100}")

print(f"\n Total points dans la collection : {len(points)}")
print(f"\n Dashboard Qdrant : {QDRANT_URL}/dashboard")
