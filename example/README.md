# Pact

## Test the website locally

Use **two terminals**.

**Terminal 1 – backend** (must be running first):

```bash
cd backend
npm install
npm start
```

Leave it running. You should see: `Server running at http://localhost:3001`.

**Terminal 2 – frontend**:

```bash
cd frontend
npm install
npm run dev
```

Vite will print a URL, e.g. **http://localhost:5173**. Open that URL in your browser. You’ll see the PII popup first, then the chat. Messages are sent to the backend; the first reply may take a while while the Llama model loads.

**If you use the LLM (Llama) backend:** install Python deps once and add your Hugging Face token as below, then start the backend as above.

---

## Backend (Llama model – normal chat)

The backend uses the **LLM** folder’s Llama model (Hugging Face) as a normal chat: one user message in, one assistant reply out. No GPT/OpenAI, no Ollama.

1. **Python deps** (use a venv or conda):

   ```bash
   cd LLM
   pip install -r requirements.txt
   ```

   For GPU: install PyTorch with CUDA first, then the rest. Put your Hugging Face token in `LLM/hg` or `LLM/hg.txt` if the model is gated.

2. **Start the backend** (port 3001):

   ```bash
   cd backend
   npm install
   npm start
   ```

   The server keeps one Python process running (`LLM/chat_server.py`) so the model loads **once**; the first message may be slow (1–2 min), later replies are much faster.

Optional: set `PORT` via env.

## Frontend (UI)

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (e.g. **http://localhost:5173**). Chat messages go to the backend, which runs the Llama model and returns the reply.
