from flask import Flask, Response
import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Image storage directory
IMAGE_DIR = "/home/iot/app"
os.makedirs(IMAGE_DIR, exist_ok=True)

def save_image(img, prefix="image"):
    """Save image with timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{IMAGE_DIR}/{prefix}_{timestamp}.jpg"
    cv2.imwrite(filename, img)
    logger.debug(f"Saved image: {filename}")
    return filename

@app.route("/capture")
def capture():
    try:
        # Open USB camera (usually /dev/video0)
        cap = cv2.VideoCapture(0)  # 0 = /dev/video0
        if not cap.isOpened():
            logger.error("Could not open video device")
            return Response("Camera not available", status=500)

        # Give the camera time to warm up
        time.sleep(1)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.error("Failed to read from camera")
            return Response("Failed to capture frame", status=500)

        # Save raw image
        save_image(frame, prefix="raw")

        # Run YOLO detection
        results = model(frame, conf=0.25)
        annotated_img = results[0].plot()

        # Save annotated image
        save_image(annotated_img, prefix="annotated")

        # Encode annotated image
        ret, buffer = cv2.imencode(".jpg", annotated_img)
        if not ret:
            logger.error("Failed to encode image")
            return Response("Failed to encode image", status=500)

        logger.debug("Serving annotated image")
        return Response(buffer.tobytes(), mimetype="image/jpeg")

    except Exception as e:
        logger.error(f"Capture error: {e}")
        return Response(f"Error: {e}", status=500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5078)
