# app.py
import streamlit as st
import asyncio
from agents import apo_workflow # CORRECTED IMPORT: from agents (plural)
from utils import HIGH_SPEED_MODE 

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Universal Optimization Agent Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# FINAL CRITICAL FIX: CACHED FUNCTION TO ISOLATE ASYNC CALL
# ----------------------------------------------------
@st.cache_data(show_spinner=False)
def get_workflow_results(task: str):
    """
    Safely runs the asynchronous apo_workflow using asyncio.run() 
    to prevent event loop conflicts and caches the result.
    """
    # The async function is run safely within the cached thread
    return asyncio.run(apo_workflow(task))


# --- Display Configuration ---
if HIGH_SPEED_MODE:
    st.sidebar.success("✅ High-Speed Mode (Groq) is ACTIVE")
else:
    st.sidebar.warning("Slow Mode (Ollama) is Active. Set GROQ_API_KEY for speed.")

st.sidebar.markdown(
    """
    ## 🛠️ Workflow Steps
    1. User submits vague **Abstract Task**.
    2. **Optimization Agent** (LLM-1) generates a precise **Optimized Prompt**.
    3. **Execution Agent** (LLM-2) executes the optimized prompt.
    4. **Final Output** (Code/Text) is displayed.
    """
)

st.title("💡 Universal Optimization Agent")
st.caption("Demonstrating the power of Meta-Prompting for guaranteed quality AI output.")

# --- User Input ---
abstract_task = st.text_area(
    "Enter a vague or general task for the AI to solve:",
    value="write a quick function to multiply two numbers",
    height=150
)

# --- Run Button ---
if st.button("Run Optimization Workflow", type="primary"):
    if not abstract_task.strip():
        st.error("Please enter a task description.")
    else:
        # Use st.spinner to show progress while the async task runs
        with st.spinner('Running the full two-stage agent workflow...'):
            try:
                # Clear cache to force a new run
                get_workflow_results.clear()
                
                # Call the cached function which safely executes the async logic
                results = get_workflow_results(abstract_task.strip())
                
                # --- Display Results ---
                st.subheader("Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Optimization Cycle Time (PECT)", 
                        value=f"{results['execution_time_seconds']:.2f} s"
                    )
                    st.info(f"**Role Selected:** {results['role_selected']}")

                with col2:
                    st.metric(
                        label="Output Type", 
                        value=results['output_type'].upper()
                    )
                    st.info(f"**Original Task:** {results['user_task'][:50]}...")
                
                st.markdown("---")

                # Optimized Prompt Section
                st.subheader("1. Optimized Prompt (The 'Pitch' Value)")
                st.code(results['optimized_prompt'], language="markdown")
                st.markdown(
                    "This clean, precise prompt is what guarantees the high-quality final output."
                )

                # Final Output Section
                st.subheader("2. Final AI Output")
                if results['output_type'] == 'code':
                    st.code(results['final_output'], language="python")
                else:
                    st.markdown(results['final_output'])

            except Exception as e:
                # This catch handles the ConnectionError raised from agents.py 
                # (due to LLM failure) or other errors.
                st.error(f"An error occurred during execution: {e}")
                st.warning("Please ensure your LLM configuration is correct (Groq API Key or 'ollama serve' is running).")
