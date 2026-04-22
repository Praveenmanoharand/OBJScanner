from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from groq import Groq
import base64
import json
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Image Deconstructor API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

@app.get("/")
async def root():
    return {"status": "online", "message": "Aura Image Deconstructor is active."}

@app.post("/api/image/deconstruct")
async def deconstruct_image(file: UploadFile = File(...)):
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")
    
    try:
        # Read and encode image
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        prompt = """
        Analyze this image like a professional graphic designer and UI/UX expert.
        Deconstruct the image into its core elements. 
        Identify and describe:
        1. TEXT: All visible text, fonts, and hierarchy.
        2. GRAPHICS: Logos, icons, or decorative elements.
        3. SUBJECTS: Main people or objects.
        4. COLOR PALETTE: Dominant HEX codes and mood.
        5. LAYOUT: The structure and composition style.
        
        Return the result in this EXACT JSON format:
        {
            "elements": [
                {"type": "text", "content": "...", "style": "..."},
                {"type": "graphic", "description": "...", "context": "..."},
                {"type": "subject", "description": "..."},
                {"type": "palette", "colors": ["#...", "#..."]},
                {"type": "layout", "description": "..."}
            ],
            "overall_vibe": "..."
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
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        return {
            "success": True,
            "data": json.loads(completion.choices[0].message.content)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
