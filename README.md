# PACT — Personalized Assessment & Control for Trustworthy LLM Privacy

PACT is a privacy-preserving middleware layer for large language model interactions. Before your query ever reaches a cloud AI, PACT intercepts it, identifies sensitive information across five configurable categories, and replaces it with anonymized placeholders — all locally on your machine. Only the sanitized version of your prompt is sent to the cloud.

A built-in **AU-Probe** uncertainty gate inspects the sanitized prompt and warns you if too much context has been removed for the AI to give a useful response, helping you strike the right balance between privacy and utility.

---

## How It Works

1. You type a query and select which privacy categories to redact (Identity, Location, Demographics, Health, Financial).
2. Five parallel redaction modules process your query simultaneously, each masking its respective category.
3. A local LLM (Llama 3.1 via Ollama) synthesizes the redacted candidates into a single clean prompt.
4. The AU-Probe scores the final prompt — if too much information was lost, you are warned before any cloud call is made.
5. The sanitized prompt is sent to OpenAI GPT, and the response is returned to you.

Your raw query never leaves your machine.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) (local LLM runtime)
- An OpenAI API key

---

## Setup

### 1. Install Ollama

**Windows / macOS:**
Download and run the installer from [https://ollama.com/download](https://ollama.com/download).

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull the Llama 3.1 model

```bash
ollama pull llama3.1:8b
```

This downloads the model (~4.7 GB). Only required once.

### 3. Clone the repository

```bash
git clone https://github.com/AnushreeU13/IS597-Project-PACT.git
cd IS597-Project-PACT
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Running PACT

### Step 1 — Start the backend

**macOS / Linux:**
```bash
python backend/server.py
```

**Windows (PowerShell):**
```powershell
python backend/server.py
```

You should see:
```
Ollama model ready: llama3.1:8b at http://localhost:11434
Uvicorn running on http://0.0.0.0:8000
```

### Step 2 — Open the frontend

**Windows (PowerShell):**
```powershell
start frontend/index.html
```

**macOS:**
```bash
open frontend/index.html
```

Or open `frontend/index.html` directly in your browser from File Explorer / Finder.

### Step 3 — Enter your OpenAI API key

Paste your OpenAI API key (`sk-...`) into the **OpenAI API Key** field in the left sidebar. It is stored only for the duration of your browser session and is never logged or saved server-side.

---

## Privacy Categories

| Category | Examples |
|---|---|
| Identity & PII | Names, emails, phone numbers, SSNs |
| Location & Geo | Addresses, cities, GPS coordinates |
| Demographics | Nationality, ethnicity, age |
| Health & Medical | Diagnoses, medications, clinical records |
| Financial | Bank accounts, card numbers, transaction amounts |

Toggle any combination on or off before sending your query.

---

## AU-Probe Uncertainty Gate

After redaction, PACT scores the sanitized prompt for information sufficiency. If the score exceeds the uncertainty threshold (0.8), the query is held back and you receive a message explaining that too much context was removed. This prevents the cloud LLM from returning a vague or misleading answer due to an over-redacted prompt.

The score and threshold are visible in the **Pipeline** trace panel beneath each response.

---

## PDF Support

Click **Attach PDF** to extract text from a document. The extracted content is combined with your question and passed through the same privacy pipeline before being sent to GPT.

---

## Project Structure

```
├── backend/
│   └── server.py          # FastAPI backend, privacy pipeline orchestration
├── modules/
│   ├── local_llama.py     # Ollama wrapper + AU-Probe integration
│   ├── pipeline_collect.py
│   ├── identity_module.py
│   ├── modules_geo.py
│   ├── demographic_module.py
│   ├── health_module.py
│   ├── financial_detector.py
│   └── synthesis_prompt.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
└── data/
    └── au_probe/
        └── linearprobe_layer_32.pt
```

---

## Troubleshooting

**Backend fails to start** — Make sure Ollama is running. You can verify with:
```bash
ollama list
```

**"Model not found" error** — Run `ollama pull llama3.1:8b` and restart the backend.

**Slow first response** — The first request may take longer as Ollama loads the model into memory. Subsequent requests are faster.

**OpenAI errors** — Double-check your API key in the sidebar. Ensure it has available quota.
