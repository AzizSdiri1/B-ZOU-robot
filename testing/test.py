import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

def load_model(model_path):
    """Load the YOLOv8 model from the specified path."""
    return YOLO(model_path)

def process_image(model, image_path, output_dir, confidence_threshold=0.25):
    """Process a single image with the YOLOv8 model and save the results."""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Perform inference
    results = model.predict(image_path, conf=confidence_threshold)

    # Get the annotated image
    annotated_img = results[0].plot()  # Plot bounding boxes and labels

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the annotated image
    output_path = os.path.join(output_dir, f"result_{Path(image_path).name}")
    cv2.imwrite(output_path, annotated_img)
    print(f"Processed image saved to {output_path}")

    # Optionally display the image (comment out if running in a non-GUI environment)
    # cv2.imshow("Detection Results", annotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def main():
    # Path to the trained model
    model_path = "runs/detect/taco_yolov8m7/weights/best.pt"
    
    # Directory containing images to test
    input_dir = "test_images"  # Replace with your test images directory
    output_dir = "output_images"  # Directory to save results

    # Load the model
    model = load_model(model_path)

    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return

    # Process each image in the input directory
    for image_file in os.listdir(input_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_file)
            print(f"Processing {image_path}...")
            process_image(model, image_path, output_dir)

if __name__ == "__main__":
    main()