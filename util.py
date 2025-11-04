import asyncio
import os
import re
import streamlit as st # Added Streamlit import for secrets
from typing import Optional, Dict, Any, Awaitable, Callable, Tuple


# --- LLM Client Setup ---

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import ollama
except ImportError:
    ollama = None

# Configuration
MODEL_NAME = "llama3.1"
GROQ_MODEL = "moonshotai/kimi-k2-instruct-0905"

# CRITICAL FIX: Prioritize Streamlit secrets manager for deployment
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)

# Initialize Clients
if GROQ_API_KEY and Groq:
    GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
    HIGH_SPEED_MODE = True
else:
    GROQ_CLIENT = None
    HIGH_SPEED_MODE = False

# --- Core LLM Call Functions ---

def _ollama_generate(prompt: str, model: str = MODEL_NAME, temperature: float = 0.1) -> str:
    """
    Synchronous call to Ollama with GUARANTEED string return on failure.
    """
    if ollama is None:
        return "Error: Ollama library is not installed."
    
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={'temperature': temperature}
        )
        
        if isinstance(response, dict) and 'response' in response:
            return response['response']
        else:
            return "Error: Ollama server returned an invalid or empty response format."

    except Exception as e: 
        # CATCH ALL: This handles connection errors, which is key to preventing the NoneType crash.
        error_msg = str(e).splitlines()[0] if str(e) else "Unknown Ollama error."
        print(f"Ollama Call Error: {error_msg}")
        return f"Error: Failed to connect to Ollama server or model is missing. Please run 'ollama serve'. Details: {error_msg}"


async def call_llm(prompt_to_send: str, is_meta_prompt: bool = True) -> str:
    """
    Asynchronously calls the LLM, prioritizing Groq, otherwise falling back to Ollama.
    """
    max_tokens = 500 if is_meta_prompt else 700
    
    if HIGH_SPEED_MODE:
        try:
            # Groq call (high-speed)
            response = await asyncio.to_thread(
                GROQ_CLIENT.chat.completions.create,
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt_to_send}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as ex: 
            # Catch Groq API/network errors and fall back to Ollama.
            print(f"API error (Groq), falling back to ollama: {ex}")
            return await asyncio.to_thread(_ollama_generate, prompt_to_send)
            
    # Ollama call (Executed if HIGH_SPEED_MODE is False)
    return await asyncio.to_thread(_ollama_generate, prompt_to_send)

# --- Prompt Construction and Parsing ---

# (build_meta_instruction remains the same)
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
