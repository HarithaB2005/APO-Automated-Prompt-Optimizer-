import os
import time
import json
import asyncio
import requests
import streamlit as st
from typing import Optional, Any, Dict

# --- Configuration & Initialization ---
# Prioritize reading the API key from Streamlit secrets
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # Fallback for local development
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)

# Set high speed mode flag
HIGH_SPEED_MODE = bool(GROQ_API_KEY)

# --- LLM Client Setup ---

# Groq Model (Fastest)
GROQ_MODEL = "mixtral-8x7b-32768"

# Ollama Model (Local Fallback)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3") # Use a fast local model

# --- Asynchronous Ollama Call (for fallback) ---

async def _ollama_generate(prompt: str) -> Optional[str]:
    """Synchronous function to call Ollama, wrapped for async."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    
    try:
        # Use requests, but run it in a thread to avoid blocking the event loop
        response = requests.post(OLLAMA_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        # Ollama returns a JSON response; extract the content
        data = response.json()
        return data.get("response", "").strip()
    
    except requests.exceptions.RequestException as e:
        print(f"Ollama Request Error: {e}")
        return None
    except json.JSONDecodeError:
        print("Ollama Response Error: Could not decode JSON.")
        return None


# --- Main Universal LLM Caller ---

async def call_llm(prompt: str, is_fast_mode: bool = HIGH_SPEED_MODE, max_retries: int = 2) -> Optional[str]:
    """
    Calls the LLM (Groq or Ollama) with retry logic.
    Returns the generated text or None on failure.
    """
    if is_fast_mode:
        # --- Groq Call (High Speed) ---
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "stream": False,
        }

        for attempt in range(max_retries):
            try:
                # Run the synchronous request in a separate thread
                response = await asyncio.to_thread(requests.post, url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # CRITICAL FIX: Ensure content is not None before returning
                if content is not None:
                    return content.strip()
                
            except requests.exceptions.RequestException as e:
                print(f"Groq API Error on attempt {attempt+1}: {e}")
            except (KeyError, IndexError) as e:
                print(f"Groq Response Structure Error on attempt {attempt+1}: {e}")
            
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)

        # If Groq fails after retries, we now automatically fall back to Ollama
        print(f"Groq failed after {max_retries} retries. Falling back to Ollama...")
        # Note: Ollama call is still wrapped in asyncio.to_thread inside _ollama_generate
        return await _ollama_generate(prompt)

    else:
        # --- Ollama Call (Local Fallback) ---
        print("Using Ollama (Local Fallback)...")
        return await _ollama_generate(prompt)
