import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# This is the path to the folder where you have stored your content
DATA_PATH = "travel_guides/"
# This is the path to the directory where we will store our vector database
DB_PATH = "db"

# --- 1. Load documents with robust error handling ---
print("Loading documents from multiple formats (PDFs, TXTs)...")

documents = []
# Iterate through all files in the directory
for file_path in os.listdir(DATA_PATH):
    full_path = os.path.join(DATA_PATH, file_path)
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
            documents.extend(loader.load())
        elif file_path.endswith(".txt"):
            loader = TextLoader(full_path)
            documents.extend(loader.load())
    except Exception as e:
        # If loading a file fails, print an error and skip it
        print(f"Error loading file: {full_path}")
        print(f"Reason: {e}\n")
        continue

print(f"Successfully loaded {len(documents)} document(s) in total.")

# --- 2. Split the documents into chunks ---
if documents:
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # --- 3. Create embeddings and store in a local database ---
    print("Creating embeddings and storing in the database... This may take a moment.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    # This creates a persistent, file-based database in the 'db' folder.
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_PATH # This tells Chroma where to save the files
    )

    print("-------------------")
    print("✅ Data ingestion complete! The 'memory' has been created.")
    print(f"✅ The vector database is now stored in the '{DB_PATH}/' directory.")
    print("-------------------")
else:
    print("No documents were successfully loaded. Skipping database creation.")

