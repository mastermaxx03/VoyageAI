import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
# We'll use the 'googlesearch' library to perform automated searches.
from googlesearch import search

# --- CONFIGURATION ---

# We've removed the hardcoded list. The agent will ask you for topics.
NUM_RESULTS_PER_QUERY = 5 
DOWNLOAD_FOLDER = "travel_guides/"


# --- HELPER FUNCTIONS (No changes here) ---

def save_webpage_as_text(url, folder):
    """Fetches a webpage, extracts its text, and saves it as a .txt file."""
    try:
        print(f"-> Scraping article: {url}")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string if soup.title else urlparse(url).path.strip('/')[-20:]
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
        file_name = os.path.join(folder, f"{safe_title}.txt")

        if os.path.exists(file_name):
            print(f"  - Already exists. Skipping.")
            return

        text_content = ""
        for p in soup.find_all('p'):
            text_content += p.get_text(separator=" ", strip=True) + "\n\n"

        if text_content:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"  - Successfully saved as '{file_name}'")
        else:
            print(f"  - No paragraph text found. Skipping.")

    except Exception as e:
        print(f"  - Failed to process webpage {url}. Reason: {e}")

def save_pdf(url, folder):
    """Downloads a PDF from a URL and saves it."""
    try:
        print(f"-> Downloading PDF: {url}")
        file_name = os.path.join(folder, os.path.basename(urlparse(url).path))

        if os.path.exists(file_name):
            print(f"  - Already exists. Skipping.")
            return

        pdf_response = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        pdf_response.raise_for_status()

        with open(file_name, 'wb') as f:
            for chunk in pdf_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  - Successfully downloaded '{file_name}'")

    except Exception as e:
        print(f"  - Failed to download {url}. Reason: {e}")


# --- MAIN INTERACTIVE LOOP ---
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

print("🚀 Starting Interactive Research Agent...")
print("Enter a research topic (e.g., 'hidden gems of Rajasthan') or type 'done' to finish.")

while True:
    # Ask the user for a search query
    query = input("\n> Research Topic: ")

    # Check if the user wants to exit
    if query.lower() in ['done', 'exit', 'quit']:
        break
    
    if not query.strip():
        continue

    print(f"\n🔍 Searching for: '{query}'")
    
    try:
        # Use the search library to find URLs
        found_urls = list(search(query, num_results=NUM_RESULTS_PER_QUERY))
        print(f"  - Found {len(found_urls)} potential links.")

        for url in found_urls:
            if not url.strip():
                continue
            
            if url.lower().endswith('.pdf'):
                save_pdf(url.strip(), DOWNLOAD_FOLDER)
            else:
                save_webpage_as_text(url.strip(), DOWNLOAD_FOLDER)
                
    except Exception as e:
        print(f"An error occurred during search: {e}")


print("\n\n--- ✅ Research and scraping process complete. ---")
print("--- Run ingest_data.py next to update the AI's memory. ---")
