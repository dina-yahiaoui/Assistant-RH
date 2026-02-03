import os
import getpass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==== Config OpenRouter ====
# Option 1 : définir la clé en dur (pour tests locaux, pas en prod)
# os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-..."

# Option 2 (recommandé) : si pas définie, la demander au lancement
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Enter OpenRouter API key: ")

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ==== Modèle de chat via OpenRouter ====
llm = ChatOpenAI(
    model="openai/gpt-4.1-mini",           # choisis un modèle dispo sur OpenRouter
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

# ==== Embeddings via OpenRouter ====
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",  # modèle d'embedding via OpenRouter
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

# ==== Vector store Chroma ====
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# ==== Charger le CSV ====
loader = CSVLoader(
    file_path="C:\\Users\\pierr\\Desktop\\Assistant-RH\\data\\cv\\resumes.csv",
    encoding="utf-8",
)
docs = loader.load()

print(f"Number of CSV rows (docs): {len(docs)}")

# ==== Compter les caractères du document brut ====
total_chars_raw = sum(len(d.page_content) for d in docs)
print(f"Total characters in raw docs: {total_chars_raw}")

# ==== Splitter en chunks ====
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
all_splits = text_splitter.split_documents(docs)

# ==== Compter les caractères après split ====
total_chars_chunks = sum(len(d.page_content) for d in all_splits)
print(f"Number of chunks: {len(all_splits)}")
print(f"Total characters in chunks: {total_chars_chunks}")

# ==== Ajouter dans Chroma ====
vector_store.add_documents(documents=all_splits)
print("Documents ajoutés dans Chroma avec embeddings via OpenRouter.")
