import os
import base64
import uuid
import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import motor.motor_asyncio
import uvicorn
import certifi
from groq import Groq
from dotenv import load_dotenv

# Fix SSL certificate issues on Windows
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load environment variables
load_dotenv()

# Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://...")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ar_scanner")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq Client
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    print("WARNING: GROQ_API_KEY not found in environment variables.")

# Initialize MongoDB
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = client[DATABASE_NAME]
history_collection = db.scan_history

app = FastAPI(title="OBJ AR Object Scanner API")

# CORS Configuration
# Standard local dev servers and production domains
origins = [
    "http://127.0.0.1:5500",    # VS Code Live Server
    "http://localhost:5500",
    "http://localhost:3000",
    "*",                         # Allow all for flexibility (restrict in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (Frontend)
# Assuming index.html is in the parent directory of main.py
frontend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Models
class ScanRequest(BaseModel):
    image: str  # base64 string

class ScanResponse(BaseModel):
    id: str
    label: str
    summary: str
    timestamp: str

# Helper: Fetch Wikipedia Summary
async def get_wikipedia_summary(query: str) -> str:
    # Quick sanitize
    query = query.strip().split('\n')[0]
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    # Use short timeout for Vercel
    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                extract = data.get("extract", "")
                if not extract: return ""
                # Split by period and take first two sentences
                sentences = extract.split(". ")
                if len(sentences) > 2:
                    return ". ".join(sentences[:2]) + "."
                return extract
            return ""
        except Exception as e:
            print(f"Wikipedia Timeout/Error: {e}")
            return ""

# Endpoints
@app.get("/")
@app.get("/api")
async def root():
    # Serve the main index.html file
    index_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "status": "online",
        "message": "Aura Life-Hack Lens API is active.",
        "version": "2.0.0"
    }

@app.post("/api/scan/analyze")
@app.post("/scan/analyze") # Support both for Vercel routing flexibility
async def analyze_image(request: ScanRequest):
    # Check for API client
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing in server environment.")
    
    try:
        # 1. AI Analysis with Gemini (Pivot to Life-Hacks)
        image_data = request.image
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        
        try:
            print(f"DEBUG: Starting Life-Hack Analysis...")
            prompt = """
            You are the 'Aura Life-Hack Lens'. Analyze this image and find the most interesting/prominent object.
            
            1. Identify the object.
            2. Provide a 'Life Hack': A secret, useful, or creative trick/tip for this object that most people don't know.
            3. Provide a 'Mind-Blowing Fact': A fascinating piece of history or science about this object.
            
            Format the response in this EXACT JSON:
            {
                "label": "Object Name",
                "summary": "💡 LIFE HACK: [The hack]\\n\\n✨ DID YOU KNOW? [The fact]"
            }
            """
            
            completion = groq_client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            import json
            analysis_result = json.loads(completion.choices[0].message.content)
            label = analysis_result.get("label", "Unknown Object")
            summary = analysis_result.get("summary", "No detailed analysis available.")
            
            print(f"DEBUG: AI identified: {label}")
            print(f"DEBUG: AI summary: {summary}")
        except Exception as e:
            print(f"CRITICAL ERROR (Groq): {str(e)}")
            # Fallback to simple label if JSON fails
            label = "Detection Active"
            summary = "The AI is currently analyzing the scene. Please ensure good lighting."


        # 2. Fetch Wikipedia Metadata as secondary knowledge
        wiki_summary = await get_wikipedia_summary(label)
        if wiki_summary and "could not be retrieved" not in wiki_summary:
            summary = f"{summary} \n\nKnowledge: {wiki_summary}"

        # 3. Database Caching
        scan_id = str(uuid.uuid4())
        timestamp = datetime.datetime.utcnow().isoformat()
        
        scan_record = {
            "scan_id": scan_id,
            "label": label,
            "summary": summary,
            "timestamp": timestamp
        }
        
        await history_collection.insert_one(scan_record)

        return {
            "success": True,
            "data": {
                "id": scan_id,
                "label": label,
                "summary": summary
            }
        }

    except Exception as e:
        print(f"General Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history():
    try:
        cursor = history_collection.find().sort("timestamp", -1).limit(50)
        history = []
        async for document in cursor:
            history.append({
                "id": document["scan_id"],
                "label": document["label"],
                "summary": document["summary"],
                "timestamp": document["timestamp"]
            })
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch history.")

def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == "__main__":
    local_ip = get_local_ip()
    print("\n" + "="*50)
    print("  OBJ AR SCANNER - ACTIVE")
    print("="*50)
    print(f"  COMPUTER: http://127.0.0.1:8000")
    print(f"  MOBILE:   http://{local_ip}:8000")
    print("="*50 + "\n")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
