import streamlit as st
import os
from dotenv import load_dotenv

# --- Core AI Libraries ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load API keys from .env file
load_dotenv()

# --- CONFIGURATION ---
# 1. POINT TO THE NEW, HIGH-QUALITY DATABASE
DB_PATH = "db_markdown"

# --- MAIN APP UI ---

st.set_page_config(layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["AI Travel Planner", "Update Knowledge Base"])

if page == "AI Travel Planner":
    st.title("✈️ VoyageAI: Your Personal Travel Planner")
    st.write("Ask me a question about the destinations in my knowledge base, and I'll create a plan for you!")

    # Check if the database exists before creating the RAG chain
    if not os.path.exists(DB_PATH):
        st.error(f"Error: The database at '{DB_PATH}' was not found.")
        st.warning("Please run the `intelligent_agent.py` script from your terminal to create the knowledge base.")
    else:
        # --- RAG CHAIN SETUP ---
        try:
            # 2. USE THE CORRECT, WORKING AI MODEL
            llm = ChatGroq(model_name=os.getenv("GROQ_MODEL_NAME", "meta-llama/llama-4-maverick-17b-128e-instruct"), temperature=0.7)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
            vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

            system_prompt = (
                "You are an expert travel assistant. Your task is to answer the user's question "
                "based ONLY on the context provided from the travel guides and articles. "
                "Synthesize the information into a helpful, coherent answer. "
                "If the information is not in the context, say 'I'm sorry, that information is not available in my current knowledge base.'\n\n"
                "Context: {context}"
            )
            prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # --- CHAT UI ---
            user_query = st.text_input("For example: 'Give me a 3-day itinerary for offbeat places in Rajasthan'")
            if user_query:
                with st.spinner("Researching..."):
                    response = rag_chain.invoke({"input": user_query})
                    st.markdown("### Here's my suggestion:")
                    st.write(response["answer"])

        except Exception as e:
            st.error(f"An error occurred while setting up the AI. Please check your API keys and model names.")
            st.error(f"Details: {e}")


elif page == "Update Knowledge Base":
    st.title("🧠 Update Knowledge Base")
    st.info("This application's knowledge comes from the `intelligent_agent.py` script.")
    
    st.markdown("""
    To add new information or update the AI's memory, you need to run the agent from your terminal.

    **Instructions:**
    1.  Open your terminal.
    2.  Make sure your virtual environment is activated:
        ```bash
        source venv/bin/activate
        ```
    3.  Run the intelligent agent script:
        ```bash
        python intelligent_agent.py
        ```
    4.  After the agent finishes, run the ingestion script to update the database:
         ```bash
        python ingest_data.py
        ```
    5.  Once complete, come back here and refresh the app. The AI will now have the new knowledge.
    """)