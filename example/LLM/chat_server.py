"""
Long-running chat server: load the model once, then serve requests via stdin/stdout.
Node backend spawns this process once and reuses it so the model stays in memory.
Protocol: one JSON object per line. First we print "READY\n", then for each line
we read from stdin: {"message": "..."} -> print {"response": "..."}.
"""
import json
import sys
import io

# Ensure this directory is on path
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent)

# Capture prints during model load so only READY goes to stdout
_real_stdout = sys.stdout
sys.stdout = io.StringIO()

try:
    from llama_model import load_model, chat
    load_model()
finally:
    sys.stdout = _real_stdout

# Signal we're ready
print("READY", flush=True)

# Process one JSON line per request
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
        msg = (data.get("message") or "").strip()
        reply = chat(msg) if msg else ""
        print(json.dumps({"response": reply}), flush=True)
    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
