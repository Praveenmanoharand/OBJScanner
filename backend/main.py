from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import base64
import json
from groq import Groq
from dotenv import load_dotenv
import uvicorn
from PIL import Image
import io

# Load environment variables
load_dotenv()

app = FastAPI(title="Magic Layer Separator API")

# CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Storage
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Initialize Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

@app.get("/")
async def root():
    return {"status": "online", "message": "Cyber-Corporate Magic Separator API is ready."}

@app.post("/api/magic/extract")
async def extract_layers(file: UploadFile = File(...)):
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")

    try:
        # 1. Process Upload
        file_id = str(uuid.uuid4())
        contents = await file.read()
        
        # Open image to get dimensions
        img = Image.open(io.BytesIO(contents))
        width, height = img.size
        
        # Save original
        original_filename = f"{file_id}_original.png"
        original_path = os.path.join(UPLOAD_DIR, original_filename)
        img.save(original_path)

        # 2. AI Analysis (Simulating SAM + OCR with Llama 3.2 Vision)
        # We ask Llama to identify objects and text with coordinates [0-1000 scale]
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        prompt = f"""
        You are an AI Image Segmenter. Deconstruct this image into layers.
        Identify:
        1. Foreground Objects (People, items, etc.)
        2. Text Blocks (Headlines, body text)
        3. Background context.

        For each element, provide:
        - type: 'object' or 'text'
        - content: (the text string if it's text)
        - bbox: [ymin, xmin, ymax, xmax] in 0-1000 scale.
        - description: (briefly what it is)

        Dimensions: {width}x{height}

        Return the result in this EXACT JSON:
        {{
            "layers": [
                {{ "type": "object", "bbox": [0,0,0,0], "description": "..." }},
                {{ "type": "text", "bbox": [0,0,0,0], "content": "...", "fontSize": 24 }}
            ],
            "background_description": "..."
        }}
        """

        completion = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {{
                    "role": "user",
                    "content": [
                        {{"type": "text", "text": prompt}},
                        {{
                            "type": "image_url",
                            "image_url": {{"url": f"data:image/jpeg;base64,{image_base64}"}},
                        }},
                    ],
                }}
            ],
            temperature=0.1,
            response_format={{"type": "json_object"}}
        )

        analysis = json.loads(completion.choices[0].message.content)

        # 3. Prepare Response (URLs and Coords)
        # In a real SAM setup, we would cut the images here. 
        # For this 'Magic' prototype, we send the original URL and coordinates.
        # The frontend will use Canvas to 'clip' these areas dynamically.

        return {
            "success": True,
            "original_url": f"/uploads/{original_filename}",
            "width": width,
            "height": height,
            "layers": analysis.get("layers", []),
            "background_vibe": analysis.get("background_description", "")
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
