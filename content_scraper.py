import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# --- CONFIGURATION ---
# A list of URLs to scrape. Can be webpages with articles or direct PDF links.
# I have replaced the old, dead links with new, working examples.
URLS_TO_SCRAPE = ["https://www.euttaranchal.com/tourism/best-tourist-places-of-uttarakhand.php",
    "https://www.trawell.in/uttarakhand/best-places-to-visit",
    "https://www.incredibleindia.org/content/incredible-india-v2/en/destinations/delhi/experience-delhi.html", # A working article link
    "https://www.incredibleindia.org/content/dam/incredible-india-v2/pdf/e-brochures/wildlife/wildlife-e-brochure-english.pdf" # A working PDF link
    # Your main task is to find more high-quality, working URLs to add here.
]

# The folder where we will save the downloaded content.
DOWNLOAD_FOLDER = "travel_guides/"


# --- SCRIPT LOGIC ---

def save_webpage_as_text(url, folder):
    """Fetches a webpage, extracts its text, and saves it as a .txt file."""
    try:
        print(f"Scraping webpage: {url}")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get the title for the filename, making it filesystem-safe
        title = soup.title.string if soup.title else "scraped_page"
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
        file_name = os.path.join(folder, f"{safe_title}.txt")

        # Extract all paragraph text
        text_content = ""
        for p in soup.find_all('p'):
            text_content += p.get_text(separator=" ", strip=True) + "\n\n"

        if text_content:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"Successfully saved article as '{file_name}'")
        else:
            print(f"No paragraph text found on {url}. Skipping.")

    except requests.exceptions.RequestException as e:
        print(f"Failed to process webpage {url}. Reason: {e}")

def save_pdf(url, folder):
    """Downloads a PDF from a URL and saves it."""
    try:
        print(f"Downloading PDF: {url}")
        # Get the last part of the URL path as the filename
        file_name = os.path.join(folder, os.path.basename(urlparse(url).path))

        if os.path.exists(file_name):
            print(f"'{file_name}' already exists. Skipping.")
            return

        pdf_response = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
        pdf_response.raise_for_status()

        with open(file_name, 'wb') as f:
            for chunk in pdf_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded '{file_name}'")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}. Reason: {e}")


# --- Main execution ---
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

for url in URLS_TO_SCRAPE:
    if not url.strip(): # Skip empty lines
        continue
    if url.lower().endswith('.pdf'):
        save_pdf(url.strip(), DOWNLOAD_FOLDER)
    else:
        save_webpage_as_text(url.strip(), DOWNLOAD_FOLDER)

print("\n--- Scraping process complete. ---")
