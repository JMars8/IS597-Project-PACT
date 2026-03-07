"""
Chat runner: read user message from stdin, print JSON response to stdout.
Used by the Node backend to get Llama model replies.
"""
import io
import json
import sys

# Ensure LLM folder is on path when run from project root or backend
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent)

from llama_model import chat

def main():
    message = sys.stdin.read().strip()
    if not message:
        print(json.dumps({"error": "Empty message"}))
        sys.exit(1)

    # Capture progress prints from llama_model so only JSON goes to stdout
    real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        reply = chat(message)
        sys.stdout = real_stdout
        print(json.dumps({"response": reply or ""}))
    except Exception as e:
        sys.stdout = real_stdout
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
