import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# This line finds and loads your secret key from the .env file
load_dotenv()
print("✅ API Key Loaded from .env file")

# Initialize the AI Model (the "Brain")
# We are using Llama 3, one of the best open models available.
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.7 # A higher temperature means more creative responses
)
print("✅ Groq LLM Initialized (Llama 3)")

# Define our question, which we'll send to the AI
# Since we are in Lucknow, let's ask a local question!
prompt = "I am in Lucknow. What is the historical significance of Rumi Darwaza?"
print(f"\n▶️ Sending Prompt: {prompt}")

# This is the line that sends our prompt to the AI and waits for a response
response = llm.invoke(prompt)

# Finally, print the AI's response to the screen
print("\n--- AI RESPONSE ---")
print(response.content)
print("-------------------")