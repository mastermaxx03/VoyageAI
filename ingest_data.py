import chromadb
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# This is the path to the folder where you have stored your travel guide PDF
DATA_PATH = "travel_guides/"
# This is the path to the directory where we will store our vector database
DB_PATH = "db"

# --- 1. Load the documents ---
print("Loading documents...")
# The DirectoryLoader will find and load all PDF files in the specified folder
loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} document(s).")


# --- 2. Split the documents into chunks ---
print("Splitting documents into chunks...")
# We split the loaded documents into smaller chunks to make them easier to process
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")


# --- 3. Create embeddings and store in ChromaDB ---
print("Creating embeddings and storing in the database... This may take a moment.")
# We use a powerful sentence transformer model to create numerical representations (embeddings) of our text chunks
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# We need to explicitly create a client to connect to the running ChromaDB server.
# This is the section that fixes the previous error.
client = chromadb.HttpClient(host='localhost', port=8000)

# Now, we create the Chroma vector store, telling it to use our new client
# and a collection name for our data.
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    client=client, # Use the new client
    collection_name="travel_guides_collection", # Give a name to our collection
    persist_directory=DB_PATH # The directory to save the data
)
print("-------------------")
print("✅ Data ingestion complete! The 'memory' has been created.")
print(f"✅ The vector database is stored in the '{DB_PATH}/' directory.")
print("-------------------")
