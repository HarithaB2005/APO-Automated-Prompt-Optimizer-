# cli.py
import asyncio
import sys
from agents import apo_workflow

def main():
    """
    Handles command-line input and runs the asynchronous workflow.
    """
    print("Enter any vague or general request (finish with Ctrl+D [Linux/Mac] or Ctrl+Z then Enter [Windows]):")
    
    # Read multiline input from standard input until EOF
    lines = sys.stdin.readlines()
    abstract_task = ''.join(lines).strip()

    if not abstract_task:
        print("No task entered. Exiting.")
        return

    # Run the main asynchronous workflow
    try:
        asyncio.run(apo_workflow(abstract_task))
    except ImportError as e:
        print(f"\n--- Dependency Error ---")
        print(f"Failed to run due to missing dependency: {e}")
        print("Please ensure 'groq' and 'ollama' libraries are installed if you intend to use them.")
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(e)

if __name__ == "__main__":
    main()