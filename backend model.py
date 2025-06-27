from flask import Flask, request, render_template, send_file
import cv2
import os
from ultralytics import YOLO
from pathlib import Path
import uuid

app = Flask(__name__)

# Configuration
MODEL_PATH = "best.pt"
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/output"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process image with YOLO
        results = model.predict(filepath, conf=0.25)
        annotated_img = results[0].plot()
        
        # Save output
        output_filename = f"result_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, annotated_img)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('index.html', result_image=output_filename)
    
    return render_template('index.html', error="Invalid file format")

if __name__ == '__main__':
    app.run(debug=True)