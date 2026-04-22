from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import base64
import json
import uvicorn
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

app = FastAPI(title="Magic Layer Separator API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Directory for storing processed layers (In a real app, use S3/Cloudinary)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"status": "online", "message": "Magic Layer Separator API is active."}

@app.post("/api/magic/extract")
async def extract_layers(file: UploadFile = File(...)):
    """
    Simulated AI Pipeline: 
    1. SAM (Segment Anything) for Object Extraction
    2. Tesseract for OCR Text Extraction
    3. LaMa for Background Inpainting
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")
    
    try:
        # 1. Read Uploaded Image
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # 2. Call Vision AI to identify layers and coordinates
        # In a production environment, you would run SAM and OCR here.
        # For this demo, we use Llama 3.2 Vision to 'deconstruct' the layout.
        prompt = """
        Analyze this image for layer separation. 
        1. Identify main foreground objects and their bounding boxes [top, left, width, height] in percentages.
        2. Identify all text blocks and their bounding boxes [top, left, width, height], content, and font size.
        3. Describe the background style.
        
        Return the result in this EXACT JSON format:
        {
            "background": {"style": "...", "description": "..."},
            "objects": [
                {"id": "obj1", "label": "...", "bbox": [y, x, w, h]}
            ],
            "text_layers": [
                {"id": "txt1", "text": "...", "bbox": [y, x, w, h], "fontSize": 24}
            ]
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
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(completion.choices[0].message.content)
        
        # 3. Formulate the response
        # In a real SAM/LaMa pipeline, you'd return URLs to the actual transparent PNGs.
        # Here we return the analysis so the frontend can 'simulate' the separation on canvas.
        return {
            "success": True,
            "session_id": str(uuid.uuid4()),
            "original_image": f"data:image/jpeg;base64,{image_base64}",
            "layers": analysis
        }
        
    except Exception as e:
        print(f"Error in magic extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
