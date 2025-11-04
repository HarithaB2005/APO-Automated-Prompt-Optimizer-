# agents.py
import re
import time
from utils import call_llm, build_meta_instruction

# NOTE: You MUST have the two functions below defined in this file (agents.py) 
# or imported from another file:
# async def generate_optimized_prompt(abstract_task: str, target_code: str) -> str:
# async def execute_optimized_prompt(optimized_prompt: str, target_code: str) -> str:

# Placeholder functions assumed to exist:
async def generate_optimized_prompt(abstract_task: str, target_code: str) -> str:
    meta_prompt = build_meta_instruction(abstract_task, target_code)
    # The first LLM call for optimization
    return await call_llm(meta_prompt, is_meta_prompt=True) 

async def execute_optimized_prompt(optimized_prompt: str, target_code: str) -> str:
    # The second LLM call for execution
    return await call_llm(optimized_prompt, is_meta_prompt=False)


async def apo_workflow(abstract_task: str) -> dict:
    """
    The main workflow: generate optimized prompt -> execute it -> returns results.
    """
    start_time = time.time()
    
    # 1. Generate Optimized Prompt
    optimized_prompt = await generate_optimized_prompt(abstract_task, abstract_task)
    
    # 2. Check for LLM connection failure (Step 1)
    if optimized_prompt.startswith("Error:"):
        # Raise a fatal error for Streamlit to catch if LLM connection failed
        raise ConnectionError(optimized_prompt) 
    
    # 3. Parse and Clean Prompt for display
    role_match = re.search(r"ROLE:? ?([A-Za-z0-9 ,\-]*)", optimized_prompt)
    chosen_role = role_match.group(1).strip() if role_match else "N/A"
    cleaned_prompt = re.sub(r"ROLE:? ?[A-Za-z0-9 ,\-]*\n?", "", optimized_prompt).strip()
    
    # 4. Execute Optimized Prompt
    optimized_output = await execute_optimized_prompt(optimized_prompt, abstract_task)

    # 5. Check for LLM connection failure (Step 2)
    if optimized_output.startswith("Error:"):
        raise ConnectionError(optimized_output)
    
    # 6. Extract Final Output
    code_match = re.search(r"```(.*?)\n(.*?)```", optimized_output, re.DOTALL)
    
    if code_match:
        final_output = code_match.group(2).strip()
        output_type = "code"
    else:
        final_output = optimized_output.strip()
        output_type = "text"
        
    # 7. Calculate Metrics and Return
    pect = round(time.time() - start_time, 2)
    
    return {
        "user_task": abstract_task,
        "role_selected": chosen_role,
        "optimized_prompt": cleaned_prompt,
        "final_output": final_output,
        "output_type": output_type,
        "execution_time_seconds": pect
    }