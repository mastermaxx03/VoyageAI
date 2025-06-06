import streamlit as st
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. SETUP - This part connects to our Brain and Memory ---

# Initialize the LLM from Groq (The "Brain")
# This is the same as in Phase 1
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.7
)

# Set up the connection to our vector database (The "Memory")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = chromadb.HttpClient(host='localhost', port=8000)

# Load the existing vector store
vectorstore = Chroma(
    client=client,
    collection_name="travel_guides_collection",
    embedding_function=embeddings
)

# Create a retriever, which is a tool to search our database
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # "k: 3" means it will find the top 3 most relevant chunks

# --- 2. THE RAG CHAIN - This part defines the agent's workflow ---

# This is the prompt template. It tells the AI how to behave.
# It has placeholders for {context} (from our database) and {input} (from the user).
system_prompt = (
    "You are an expert travel assistant. Your task is to answer the user's question "
    "based ONLY on the context provided from the travel guide. "
    "Do not use your own general knowledge. If the information is not in the context, "
    "say 'I'm sorry, that information is not available in the travel guide.'\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# This chain takes the user's question and the retrieved documents and stuffs them into the prompt.
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# This is the final chain. It combines the retriever and the question-answer chain.
# It automatically performs the search (retrieve) and then generates the answer (generate).
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# --- 3. THE APP - This is the Streamlit user interface ---

st.title("✈️ VoyageAI: Your Personal Travel Planner")
st.write("Ask me a question about the destinations in my travel guide, and I'll create a plan for you!")

# Get user input from a text box
user_query = st.text_input("For example: 'What are the best street food places to visit in Lucknow?'")

# If the user has typed something, run the AI
if user_query:
    with st.spinner("Thinking..."):
        # Send the user's query to our RAG chain
        response = rag_chain.invoke({"input": user_query})
        
        # Display the answer
        st.markdown("### Here is my suggestion:")
        st.write(response["answer"])
