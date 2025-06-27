import sys
from ultralytics import YOLO

if len(sys.argv) < 2:
    print("Usage: detect <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
model = YOLO("yolov8n.pt")
model.predict(source=image_path, save=True)
