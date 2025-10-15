import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# LangChain can use TextLoader for markdown files
from langchain_community.document_loaders import TextLoader

# UPDATE THIS PATH to your new markdown folder
DATA_PATH = "travel_guides_markdown/"
DB_PATH = "db_markdown" # Use a new DB path to keep it separate

# --- 1. Load documents ---
print("Loading markdown documents...")
documents = []
for file_path in os.listdir(DATA_PATH):
    if file_path.endswith(".md"):
        full_path = os.path.join(DATA_PATH, file_path)
        try:
            loader = TextLoader(full_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading file: {full_path}, Reason: {e}")
            continue

print(f"Successfully loaded {len(documents)} document(s).")

# --- 2. Split, Embed, and Store (This part remains the same) ---
if documents:
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("Creating embeddings and storing in the new database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"✅ Data ingestion complete! New database is ready in '{DB_PATH}/'.")
else:
    print("No markdown documents found to process.")