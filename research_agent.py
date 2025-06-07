import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
# We'll use the 'googlesearch' library to perform automated searches.
# You'll need to install it first: pip install googlesearch-python
from googlesearch import search

# --- CONFIGURATION ---

# 1. DEFINE YOUR RESEARCH TOPICS
# This is where you guide the agent. It will search for these phrases.
# Be specific to find hidden gems!
SEARCH_QUERIES = [
    "underrated travel destinations in India blog",
    "offbeat places to visit in the Himalayas",
    "secret beaches of Goa travel guide",
    "hidden waterfalls in Kerala article",
    "travel guide for Northeast India PDF"
]

# 2. CONFIGURE THE AGENT
# How many search results should it check for each query?
NUM_RESULTS_PER_QUERY = 5 

# The folder where we will save the downloaded content.
DOWNLOAD_FOLDER = "travel_guides/"


# --- SCRIPT LOGIC (The parts from content_scraper.py) ---

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


# --- Main execution loop ---
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

print("🚀 Starting Automated Research Agent...")

for query in SEARCH_QUERIES:
    print(f"\n🔍 Searching for: '{query}'")
    
    # Use the search library to find URLs
    found_urls = list(search(query, num_results=NUM_RESULTS_PER_QUERY))
    
    print(f"  - Found {len(found_urls)} potential links.")

    for url in found_urls:
        if not url.strip():
            continue
        
        # Check if the URL points to a PDF
        if url.lower().endswith('.pdf'):
            save_pdf(url.strip(), DOWNLOAD_FOLDER)
        # Check for common article URL patterns
        elif "blog" in url or "guide" in url or "news" in url or ".html" in url:
            save_webpage_as_text(url.strip(), DOWNLOAD_FOLDER)
        else:
            print(f"  - Skipping non-article link: {url}")


print("\n\n--- ✅ Research and scraping process complete. ---")
print("--- Run ingest_data.py next to update the AI's memory. ---")

