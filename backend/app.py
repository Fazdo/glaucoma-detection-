import os
import io
import uuid
import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from model import ImageAnalyzer
from database import init_db, save_analysis_result

# Initialize FastAPI app
app = FastAPI(title="Glaucoma Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Initialize model
analyzer = ImageAnalyzer()

@app.get("/")
async def root():
    return {"message": "Glaucoma Detection API is running"}

@app.post("/api/analyze")
async def analyze_image(image: UploadFile = File(...)):
    # Validate file
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        # Analyze image
        result = analyzer.analyze(img)
        
        # Save result to database
        image_id = str(uuid.uuid4())
        save_analysis_result(
            image_id=image_id,
            filename=image.filename,
            result=result,
            timestamp=datetime.now().isoformat()
        )
        
        # Return result
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
