from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import base64
import json
import uvicorn
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

app = FastAPI(title="Magic Grab AI API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq Client (For metadata/text analysis)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Storage for processed layers (Static files)
STORAGE_DIR = "assets/magic_layers"
os.makedirs(STORAGE_DIR, exist_ok=True)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

@app.get("/")
async def root():
    return {"status": "online", "message": "Magic Grab AI Engine is active."}

@app.post("/api/magic/extract")
async def extract_layers(file: UploadFile = File(...)):
    """
    Magic Extraction Pipeline (SAM + OCR + LaMa)
    Note: In a production environment, this would call specialized GPU models.
    For this implementation, we provide the robust API structure and simulated high-fidelity results.
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")

    try:
        session_id = str(uuid.uuid4())
        contents = await file.read()
        
        # Save original for processing reference
        original_path = f"{STORAGE_DIR}/{session_id}_original.jpg"
        with open(original_path, "wb") as f:
            f.write(contents)

        # 1. Image Analysis (Using Groq Vision to identify potential layers)
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        prompt = """
        Analyze this image for layer separation. 
        Identify:
        1. Every distinct foreground object (Position, Name).
        2. Every text element (Exact string, Position, Approximate font size).
        3. The general background context.
        
        Return the result in this EXACT JSON format:
        {
            "objects": [{"id": 1, "label": "...", "x": 100, "y": 200, "w": 300, "h": 400}],
            "text_elements": [{"id": 1, "text": "...", "x": 50, "y": 50, "fontSize": 24, "fontFamily": "Inter"}]
        }
        """
        
        completion = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(completion.choices[0].message.content)
        
        # 2. Simulated Layer Separation (Returning the original as BG and cropped placeholders for objects)
        # In a real SAM environment, we would save individual transparent PNGs here.
        
        return {
            "success": True,
            "session_id": session_id,
            "background": f"/assets/magic_layers/{session_id}_original.jpg", # Simulated Inpainted BG
            "layers": {
                "objects": analysis.get("objects", []),
                "text": analysis.get("text_elements", [])
            }
        }

    except Exception as e:
        print(f"PIPELINE ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
