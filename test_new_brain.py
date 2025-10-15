from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
api_key = "***REMOVED***"
# --- CONFIGURATION ---
DB_PATH = "db_markdown" # Point to your NEW database

# --- RAG CHAIN SETUP ---
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.7)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an expert travel assistant. Answer the user's question "
    "based ONLY on the context provided.\n\n"
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- ASK A QUESTION ---
user_query = "What are some hidden gems in Rajasthan?"
print(f"▶️ Sending Query: {user_query}")

response = rag_chain.invoke({"input": user_query})

print("\n--- AI RESPONSE ---")
print(response["answer"])
print("-------------------")