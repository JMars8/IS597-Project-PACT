from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os

# Import our detectors
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from modules.financial_detector import FinancialDetector
from modules.geo_detector import GeoDetector

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatSettings(BaseModel):
    identity: bool
    location: bool
    demographic: bool
    health: bool
    financial: bool

class ChatRequest(BaseModel):
    query: str
    settings: ChatSettings

financial_detector = FinancialDetector()
geo_detector = GeoDetector()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    query = request.query
    sanitizations = []
    
    # 1. Privacy Processing - Financial
    if request.settings.financial:
        processed_query, financial_logs = financial_detector.detect_and_redact(query)
        query = processed_query
        sanitizations.extend(financial_logs)

    # 2. Privacy Processing - Geo (GPE, LOC, FAC)
    if request.settings.location:
        processed_query, geo_logs = geo_detector.detect_and_redact(query)
        query = processed_query
        sanitizations.extend(geo_logs)
    
    # 3. Mock LLM Synthesis (Placeholder for Llama)
    # In a real scenario, the 'query' here is already sanitized.
    response_text = f"As a privacy-focused assistant, I've received your request. "
    if sanitizations:
        response_text += f"I have successfully sanitized {len(sanitizations)} private data points. "
    
    response_text += f"<br><br><strong>Simulated Llama Response:</strong><br>I understand your query: \"{query}\". How else can I assist you with this?"

    return {
        "response": response_text,
        "sanitizations": sanitizations,
        "original_query_sanitized": query
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
