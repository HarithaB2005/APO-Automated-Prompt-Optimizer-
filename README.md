💡 Autonomous Prompt Optimization (APO) Agent Demo
The Autonomous Prompt Optimization (APO) Agent is the two-stage system that fixes AI's core reliability problem, guaranteeing high-quality, actionable output and drastically cutting the Prompt Engineering Cycle Time (PECT).This repository contains the full Streamlit application and agent workflow used to demonstrate the APO concept in live presentations.🚀 Key FeaturesTwo-Stage Meta-Prompting: An initial LLM acts as the Optimization Agent (LLM-1) to transform vague user input into a precise, guardrailed prompt before a second LLM (LLM-2) executes it.Guaranteed Consistency: Ensures outputs meet defined quality standards (e.g., proper docstrings, error handling, clean variables).Flexible LLM Backend: Supports Groq (High-Speed Mode) for rapid inference and Ollama (Local Fallback) for processing via a local server.Live Demo: A fully interactive Streamlit frontend for real-time demonstration of the APO workflow.

🛠️ Setup and Installation
Follow these steps to get a local copy running.
1. Clone the Repository:
     git clone https://github.com/HarithaB2005/APO-Automated-Prompt-Optimizer-.git
     cd APO-Automated-Prompt-Optimizer-
3. Install Dependencies:
     Install the required Python packages using the provided requirements.txt file:
           pip install -r requirements.txt
4. Configure LLM Access
   The agent requires access to either the Groq API or a local Ollama server
   A. High-Speed Mode (Groq API - Recommended)
       Set your Groq API key as an environment variable for high-speed operation:
          export GROQ_API_KEY="sk_..."
   B. Local Fallback (Ollama)
       For the local fallback to work, you must have the Ollama server running and the required model pulled:
     Start Ollama Server:
         ollama serve
      Pull Model:
        ollama pull llama3.1
6. Run the Application
   Execute the Streamlit application from your terminal:
   streamlit run app.py
The application will open in your browser, demonstrating the agent workflow.
📂 File Structure:
    File                                              Description
   app.py                        The main Streamlit frontend. Handles user input and displays                                     results. Uses @st.cache_data for safe asynchronous execution.
   agents.py                     The core agent workflow. Contains the apo_workflow logic,                                         managing the two-stage LLM calls and processing outputs.
   utils.py                       Utility functions for LLM connectivity (call_llm), environment                                   setup, and robust error handling. Guarantees string output to                                    prevent application crashes.
requirements.txt                  Lists all necessary Python dependencies (streamlit, groq,                                         ollama, etc.).

🔗 Live Demo Link
For Investors and Audience: Please visit our live, deployed version for the best experience (runs on High-Speed Mode via Groq):

[Your unique Streamlit App URL, e.g., https://your-username-apo-agent.streamlit.app]

🤝 Contribution
We welcome feedback and suggestions for improving the agent performance.
