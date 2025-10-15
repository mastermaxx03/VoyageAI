Of course. Here is a more concise README for your project.

-----

# ✈️ VoyageAI: Your Personal AI Travel Planner

VoyageAI is an intelligent, self-updating travel planner that uses a Retrieval-Augmented Generation (RAG) pipeline to provide personalized travel itineraries based on a curated knowledge base.

## ✨ Core Features

  * **AI-Powered Itinerary Generation**: Ask natural language questions and receive detailed travel plans.
  * **Intelligent Research Agent**: An autonomous agent that uses an LLM to generate its own research queries and discover new travel information online.
  * **Advanced Document Processing**: Integrates Docling to parse complex documents (PDFs, webpages) into clean markdown, ensuring high-quality data for the AI.
  * **Interactive Web UI**: A simple and clean user interface built with Streamlit for easy interaction.

## ⚙️ How It Works

1.  **Offline Pipeline**: The `intelligent_agent.py` script searches the web for travel guides, uses Docling to process them into clean markdown, and `ingest_data.py` builds a Chroma vector database from this content.
2.  **Online Querying**: The `app.py` Streamlit app loads the pre-built database. [cite\_start]When a user asks a question, it retrieves the most relevant information and uses a Groq LLM to generate a context-aware answer[cite: 16, 17, 18].

## 🛠️ Tech Stack

  * **Backend**: Python, LangChain, Streamlit
  * **AI**: Groq, HuggingFace Embeddings
  * **Data Processing**: Docling, ChromaDB, BeautifulSoup4

## 🚀 Getting Started

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/mastermaxx03/VoyageAI.git
    cd VoyageAI
    ```

2.  **Set Up Environment & Install Dependencies**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure API Keys**

      * Create a `.env` file in the root directory.
      * Add your `GROQ_API_KEY`, `DOCLING_API_KEY`, and `GROQ_MODEL_NAME`.

4.  **Build the Knowledge Base**
    Run the offline scripts to gather data and build the database.

    ```bash
    python intelligent_agent.py
    python ingest_data.py
    ```

5.  **Run the Application**

    ```bash
    streamlit run app.py
    ```
