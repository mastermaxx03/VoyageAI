import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from docling import DoclingClient
import requests
from bs4 import BeautifulSoup
from googlesearch import search 
load_dotenv()
DOWNLOAD_FOLDER = "travel_guides_markdown/"
NUM_RESULTS_PER_QUERY = 3



def save_processed_content(content_bytes, destination_path):
    """
    Takes raw content bytes, sends them to Docling for parsing,
    and saves the resulting clean markdown to a file.
    """
    try:
        print(f"->Sending content to docling for processing")
        parsed_result=docling_client.parse(content=content_bytes,
                                            output_format="markdown")
        clean_markdown=parsed_result.text
        if clean_markdown:
            with open(destination_path,'w',encoding='utf-8') as f:
                f.write(clean_markdown)
            print(f"  ✅ Successfully saved processed markdown to '{destination_path}'")
            return True
        else:
            print(f"  ⚠️ Docling returned no content. Skipping.")
            return False
        

    except Exception as e:
        print(f"  ❌ Docling processing failed. Reason: {e}")
        return False
def fetch_pdf_content(url):
    """(The Gatherer) Downloads the raw content of a PDF from a URL."""
    try:
        print(f"  -> Fetching PDF content from: {url}")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        response.raise_for_status()
        return response.content  # Returns the raw bytes of the PDF
    except Exception as e:
        print(f"  ❌ Failed to fetch PDF {url}. Reason: {e}")
        return None
    
def fetch_webpage_content(url):
    """(The Gatherer) Downloads a webpage's paragraph text as bytes."""
    try:
        print(f"  -> Fetching webpage content from: {url}")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extracts only paragraph text and converts it to bytes
        text_content = "".join(p.get_text(separator="\n", strip=True) + "\n\n" for p in soup.find_all('p'))
        return text_content.encode('utf-8')
    except Exception as e:
        print(f"  ❌ Failed to fetch webpage {url}. Reason: {e}")
        return None    
    
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.8)
docling_client = DoclingClient(api_key=os.getenv("DOCLING_API_KEY"))


print("🧠 Giving the agent its research mission...")
mission = "Find in-depth guides about hidden gems and offbeat destinations in Rajasthan"

prompt = f"""
You are a brilliant research assistant. Your mission is to generate a diverse list of 5 Google search queries to find information about: '{mission}'.

- Include queries that specifically look for PDF brochures using operators like 'filetype:pdf'.
- Create queries that target blogs and detailed travel guides.
- Return ONLY a Python-parsable list of strings.

Example format:
["query 1", "query 2", "query 3", "query 4", "query 5"]
"""

try:
    response = llm.invoke(prompt)
    search_queries = eval(response.content.strip())
    print(f"✅ Agent's plan is ready: {search_queries}")
except Exception as e:
    print(f"❌ LLM failed. Using a fallback plan. Reason: {e}")
    search_queries = ["offbeat places to visit in Rajasthan blog"]
# --- END OF MISSING BLOCK ---


# --- MAIN EXECUTION LOOP (Your existing code starts here) ---
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
print("\n🚀 Starting Automated Research...")

for query in search_queries:
    print(f"\n🔍 Searching for: '{query}'")
    try:
        found_urls = list(search(query, num_results=NUM_RESULTS_PER_QUERY))
        print(f"  - Found {len(found_urls)} potential links.")

        for url in found_urls:
            url = url.strip()
            
            # Create a simple, safe filename for the final markdown file
            safe_title = "".join(c for c in url if c.isalnum())[-60:]
            destination = os.path.join(DOWNLOAD_FOLDER, f"{safe_title}.md")

            if os.path.exists(destination):
                print(f"  - Already processed this URL. Skipping.")
                continue

            # Decide which gatherer to use
            if url.lower().endswith('.pdf'):
                raw_content = fetch_pdf_content(url)
            else:
                raw_content = fetch_webpage_content(url)

            # If the gatherer was successful, send the content to the processor
            if raw_content:
                save_processed_content(raw_content, destination)

    except Exception as e:
        print(f"  - An error occurred during the search for '{query}': {e}")

print("\n\n--- ✅ Research and processing complete. ---")    