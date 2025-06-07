import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from googlesearch import search
import time

import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
DB_PATH = "db"
DOWNLOAD_FOLDER = "travel_guides/"
NUM_RESULTS_PER_QUERY = 3  # Kept it low to make research faster for the UI

# --- HELPER FUNCTIONS for Research & Ingestion ---

def save_webpage_as_text(url, folder, status_area):
    try:
        status_area.write(f"-> Scraping article: {url}")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string if soup.title else urlparse(url).path.strip('/')[-20:]
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
        file_name = os.path.join(folder, f"{safe_title}.txt")

        if os.path.exists(file_name):
            status_area.write(f"  - Already exists: {file_name}")
            return

        text_content = "".join(p.get_text(separator=" ", strip=True) + "\n\n" for p in soup.find_all('p'))

        if text_content:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(text_content)
            status_area.write(f"  - Successfully saved: {file_name}")
        else:
            status_area.write(f"  - No paragraph text found. Skipping.")

    except Exception as e:
        status_area.write(f"  - Failed to process webpage {url}. Reason: {e}")

def save_pdf(url, folder, status_area):
    try:
        status_area.write(f"-> Downloading PDF: {url}")
        file_name = os.path.join(folder, os.path.basename(urlparse(url).path))

        if os.path.exists(file_name):
            status_area.write(f"  - Already exists: {file_name}")
            return

        pdf_response = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        pdf_response.raise_for_status()

        with open(file_name, 'wb') as f:
            for chunk in pdf_response.iter_content(chunk_size=8192):
                f.write(chunk)
        status_area.write(f"  - Successfully downloaded: {file_name}")

    except Exception as e:
        status_area.write(f"  - Failed to download {url}. Reason: {e}")

def ingest_new_content(status_area):
    """Function to run the ingestion process and show status in Streamlit."""
    status_area.write("### 🧠 Updating AI Memory...")
    
    # --- Loading ---
    status_area.write("-> Loading new documents...")
    documents = []
    for file_path in os.listdir(DOWNLOAD_FOLDER):
        full_path = os.path.join(DOWNLOAD_FOLDER, file_path)
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(full_path)
                documents.extend(loader.load())
            elif file_path.endswith(".txt"):
                loader = TextLoader(full_path)
                documents.extend(loader.load())
        except Exception as e:
            status_area.write(f"  - Error loading {file_path}: {e}")
            continue
    status_area.write(f"-> Loaded {len(documents)} documents.")

    # --- Splitting ---
    if documents:
        status_area.write("-> Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        status_area.write(f"-> Split into {len(docs)} chunks.")

        # --- Embedding & Storing ---
        status_area.write("-> Creating embeddings and storing in database (this may take a moment)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        status_area.write("✅ AI Memory Updated Successfully!")
        time.sleep(2) # Wait a moment before clearing the status
    else:
        status_area.write("No new documents to process.")

# --- MAIN APP UI ---

st.set_page_config(layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["AI Travel Planner", "Conduct New Research"])

if page == "AI Travel Planner":
    st.title("✈️ VoyageAI: Your Personal Travel Planner")
    st.write("Ask me a question about the destinations in my knowledge base, and I'll create a plan for you!")

    # Check if the database exists before creating the RAG chain
    if not os.path.exists(DB_PATH):
        st.warning("The knowledge base is empty. Please go to the 'Conduct New Research' page to add information.")
    else:
        # --- RAG CHAIN SETUP ---
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        system_prompt = (
            "You are an expert travel assistant. Your task is to answer the user's question "
            "based ONLY on the context provided from the travel guide. "
            "Do not use your own general knowledge. If the information is not in the context, "
            "say 'I'm sorry, that information is not available in the travel guide.'\n\n"
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        Youtube_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, Youtube_chain)

        # --- CHAT UI ---
        user_query = st.text_input("For example: 'What are the best street food places to visit in Lucknow?'")
        if user_query:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_query})
                st.markdown("### Here is my suggestion:")
                st.write(response["answer"])

elif page == "Conduct New Research":
    st.title("🔬 Conduct New Research")
    st.write("Add new knowledge to the AI by telling it what topics to research on the internet.")

    query = st.text_input("Enter a research topic (e.g., 'hidden gems of Rajasthan')")

    if st.button("Start Research"):
        if query:
            os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
            
            # Use an empty placeholder to show status updates in real-time
            status_area = st.empty()
            
            status_area.write(f"### 🔍 Searching for: '{query}'")
            try:
                found_urls = list(search(query, num_results=NUM_RESULTS_PER_QUERY))
                status_area.write(f"  - Found {len(found_urls)} potential links. Processing now...")
                time.sleep(2)

                for url in found_urls:
                    if url.lower().endswith('.pdf'):
                        save_pdf(url.strip(), DOWNLOAD_FOLDER, status_area)
                    else:
                        save_webpage_as_text(url.strip(), DOWNLOAD_FOLDER, status_area)
                    time.sleep(1)
                
                status_area.write("✅ Research and download complete. Now updating AI memory...")
                time.sleep(2)
                
                # After scraping, run the ingestion process
                ingest_new_content(status_area)

            except Exception as e:
                st.error(f"An error occurred during research: {e}")
        else:
            st.warning("Please enter a research topic.")
