from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import RedirectResponse
import cv2
import os
from ultralytics import YOLO
from pathlib import Path
import uuid
from auth_utils import get_current_user

router = APIRouter()

# Load YOLO model
model = YOLO("yolov8n.pt")  # Update path as needed
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPG, PNG allowed")
    
    # Save uploaded file
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Process image with YOLO
    results = model(file_path)
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    results[0].save(output_path)
    
    # Clean up uploaded file
    os.remove(file_path)
    
    return {"filename": filename}
