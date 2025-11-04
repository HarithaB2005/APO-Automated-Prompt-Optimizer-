import asyncio
import os
import re
import streamlit as st
import requests # Added requests for reliable Groq/Ollama API calls
import json
from typing import Optional, Dict, Any, Awaitable, Callable, Tuple


# --- LLM Client Setup ---

# Configuration
MODEL_NAME = "llama3.1"
GROQ_MODEL = "moonshotai/kimi-k2-instruct-0905"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

# CRITICAL FIX: Prioritize Streamlit secrets manager for deployment
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)

# Initialize Clients
# HIGH_SPEED_MODE is now purely determined by the presence of the API key,
# avoiding dependency on a successful 'from groq import Groq'
HIGH_SPEED_MODE = bool(GROQ_API_KEY)


# --- Core LLM Call Functions ---

async def _ollama_generate(prompt: str) -> str:
    """
    Synchronous function to call Ollama via requests, wrapped for async. 
    GUARANTEED to return a string (error message if failed).
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    
    try:
        # Use requests, run in a thread to avoid blocking the event loop
        response = await asyncio.to_thread(requests.post, OLLAMA_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        # Ollama returns a JSON response; extract the content
        data = response.json()
        content = data.get("response", "").strip()
        
        if content:
            return content
        else:
            return "Error: Ollama server returned an empty response."

    except requests.exceptions.RequestException as e:
        error_msg = str(e).splitlines()[0] if str(e) else "Unknown Request Error."
        print(f"Ollama Request Error: {error_msg}")
        return f"Error: Failed to connect to Ollama server. Details: {error_msg}"
    except json.JSONDecodeError:
        print("Ollama Response Error: Could not decode JSON.")
        return "Error: Ollama server returned an invalid response format (not JSON)."
    except Exception as e:
        print(f"General Ollama Error: {e}")
        return f"Error: An unexpected error occurred in Ollama call: {e}"


async def call_llm(prompt_to_send: str, is_meta_prompt: bool = True) -> str:
    """
    Asynchronously calls the LLM, prioritizing Groq (via requests), otherwise falling back to Ollama.
    GUARANTEED to return a string.
    """
    # Determine max tokens based on prompt type
    max_tokens = 500 if is_meta_prompt else 700 
    
    if HIGH_SPEED_MODE:
        # --- Groq Call (High Speed via requests) ---
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt_to_send}],
            "temperature": 0.1,
            "stream": False,
            "max_tokens": max_tokens,
        }
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Run the synchronous request in a separate thread
                response = await asyncio.to_thread(requests.post, url, headers=headers, json=payload, timeout=30)
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                
                data = response.json()
                # CRITICAL FIX: Direct access to content and validation
                content = data.get('choices', [{}])[0].get('message', {}).get('content')
                
                # Check for None and return the content safely
                if content:
                    return content.strip()
                
            except requests.exceptions.RequestException as e:
                print(f"Groq API Error on attempt {attempt+1}: {e}")
            except (KeyError, IndexError) as e:
                print(f"Groq Response Structure Error on attempt {attempt+1}: {e}")
            except Exception as e:
                # Catch general exceptions (e.g., JSONDecodeError)
                print(f"General Groq Error on attempt {attempt+1}: {e}")
            
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)

        # If Groq fails after retries, automatically fall back to Ollama
        print(f"Groq failed after {max_retries} retries. Falling back to Ollama...")
        return await _ollama_generate(prompt_to_send)

    else:
        # --- Ollama Call (Local Fallback) ---
        print("Using Ollama (Local Fallback)...")
        return await _ollama_generate(prompt_to_send)


# --- Prompt Construction and Parsing ---

def build_meta_instruction(task_description: str, target_code: str) -> str:
    """
    Constructs the detailed system instruction for the Universal Optimization Agent.
    """
    meta_instruction = f"""
You are a Universal Optimization Agent.
Rewrite the user's request so any AI assistant delivers a result that is simple, clear, and maximally user-friendly.

- For any code-related tasks, your optimized prompt MUST require:
  - Readable code with docstrings (or header comment) summarizing the overall logic.
  - Human-friendly prompt for user input (not raw 'n: ', but e.g. 'Enter N:')
  - Basic error handling for bad/edge-case input (e.g. ValueError, negative, empty)
  - Output that is easy to read and understand even for non-experts (e.g. 'The square of 7 is: 49')
  - Clean, logical variable names, and at least one inline comment explaining the key part.
  - (Bonus) Show a sample output for illustration if helpful.

- For any non-code/general tasks, ALWAYS demand:
  - Clear explanation (as a comment, docstring, or short intro)
  - If advice/instructions, steps must be actionable and immediately usable by most people
  - Never sacrifice clarity, context, or user understanding just for brevity.

**Addition:** Before formatting your answer, always pause to reflect on the user’s actual intent and context:
- If code or technical output is specifically warranted or obviously the best fit, provide it as described above.
- If the prompt is open-ended, general, or only about advice, respond only in human language—clear, actionable statements, not code or technical logic—unless the user’s intent or context changes.
- Use professional judgement, not keyword triggers, to ensure the answer feels natural and goal-oriented for the specific user and their likely scenario.
- If unsure, briefly clarify or offer a menu of helpful next actions instead of assuming their intent.

All output must be the *optimized prompt only*—no meta, no explanations, just the thing to send to the next assistant, which is concise and precise.

TASK: {task_description}
CONTEXT: {target_code}
"""
    return meta_instruction.strip()
