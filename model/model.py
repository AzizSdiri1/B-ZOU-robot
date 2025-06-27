
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import os
import yaml
from tqdm import tqdm
import glob
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
class TACOTrainer:
    def __init__(self):
        self.base_dir = Path("taco_yolo_project")
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "models"
        self.output_dir = self.base_dir / "outputs"
        self.dataset_url = "https://universe.roboflow.com/waste-detector/taco-yolo-10-class-original"
        self.create_directories()
        
        # Model configuration
        self.img_size = 640  # Reduced from 1024 to lower memory usage
        self.batch_size = 4  # Reduced from 8 to prevent OOM
        self.epochs = 50    # Reduced from 100 for faster testing
        self.pretrained_model = "yolov8m.pt"  # Switched to medium model for stability
        self.device = self.select_device()
    
    def select_device(self):
        """Select CPU or GPU with fallback to CPU if GPU fails"""
        try:
            if torch.cuda.is_available():
                device = 'cuda'
                # Test GPU memory with smaller tensor
                test_tensor = torch.randn(500, 500).to(device)
                del test_tensor
                torch.cuda.empty_cache()
                print("\n🟢 Using GPU acceleration")
                return device
        except RuntimeError as e:
            print(f"\n⚠️ GPU Error: {str(e)}")
            print("🟠 Falling back to CPU")
        
        print("\n🟡 Using CPU")
        return 'cpu'
    
    def create_directories(self):
        """Create project directory structure"""
        for directory in [self.data_dir, self.model_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def prepare_dataset(self):
        """Prepare TACO dataset in YOLO format"""
        dataset_path = self.data_dir / "taco_yolo_10_class_original"
        if not dataset_path.exists():
            print(f"\n⚠️ Dataset not found at: {dataset_path}")
            print(f"⚠️ Download from: {self.dataset_url}")
            print("⚠️ Extract into:", dataset_path)
            exit(1)
        
        # Verify dataset structure
        for split in ['train', 'valid', 'test']:
            split_path = dataset_path / split
            if not split_path.exists():
                print(f"⚠️ Missing directory: {split_path}")
                exit(1)
                
            jpg_files = list(split_path.glob("*.jpg"))
            txt_files = list(split_path.glob("*.txt"))
            print(f"  Images: {len(jpg_files)}")
            print(f"  Labels: {len(txt_files)}")
            
            missing = sum(1 for jpg in jpg_files if not jpg.with_suffix('.txt').exists())
            if missing > 0:
                print(f"  ⚠️ Warning: {missing} images missing annotations")
        
        # Read dataset configuration
        data_yaml_path = dataset_path / "data.yaml"
        if not data_yaml_path.exists():
            print(f"⚠️ Missing data.yaml at: {data_yaml_path}")
            exit(1)
        
        with open(data_yaml_path) as f:
            data_cfg = yaml.safe_load(f)
        
        print("\nDataset classes:", data_cfg.get('names', []))
        return str(data_yaml_path)

    def display_sample_images(self, num_images=20):
        """Display sample images with annotations from the training set, each in its own figure"""
        print("\n📷 Displaying sample images with annotations...")
        
        dataset_path = self.data_dir / "taco_yolo_10_class_original"
        train_images = list((dataset_path / "train").glob("*.jpg"))
        
        if len(train_images) < num_images:
            print(f"⚠️ Warning: Only {len(train_images)} images available, displaying all.")
            num_images = len(train_images)
        
        # Read class names from data.yaml
        data_yaml_path = dataset_path / "data.yaml"
        if not data_yaml_path.exists():
            print(f"⚠️ Missing data.yaml at: {data_yaml_path}")
            return
        
        with open(data_yaml_path) as f:
            data_cfg = yaml.safe_load(f)
        class_names = data_cfg.get('names', [])
        
        for i, img_path in enumerate(train_images[:num_images]):
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️ Failed to load image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
            
            # Load annotations
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                print(f"⚠️ No annotations for {img_path.name}")
            else:
                # Read bounding boxes
                with open(txt_path, 'r') as f:
                    annotations = f.readlines()
                
                h, w = img.shape[:2]
                for ann in annotations:
                    try:
                        cls_id, x_center, y_center, width, height = map(float, ann.strip().split())
                        cls_id = int(cls_id)
                        
                        # Convert YOLO format (normalized) to pixel coordinates
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                        
                        # Draw rectangle
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Add label
                        label = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ Invalid annotation format in {txt_path}: {str(e)}")
                        continue
            
            # Display image in its own figure
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f"Image {i+1}: {img_path.name}")
            plt.axis('off')
            plt.show()
            plt.close('all')  # Close to prevent memory leaks

    def train_model(self, data_yaml):
        """Train YOLOv8 model on TACO dataset"""
        print("\n🚀 Initializing YOLOv8 training...")
        
        try:
            model = YOLO(self.pretrained_model)
        except Exception as e:
            print(f"⚠️ Failed to load model: {str(e)}")
            exit(1)
        
        train_params = {
            'data': data_yaml,
            'epochs': self.epochs,
            'imgsz': self.img_size,
            'batch': self.batch_size,
            'name': 'taco_yolov8m',
            'save': True,
            'patience': 10,  # Reduced patience for faster stopping
            'degrees': 45,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.2,
            'rect': False,
            'device': self.device,
            'workers': 2 if self.device == 'cpu' else 4,
            'verbose': True
        }
        
        try:
            results = model.train(**train_params)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("\n⚠️ GPU Memory Full! Retrying with CPU...")
                train_params['device'] = 'cpu'
                train_params['batch'] = max(1, self.batch_size // 2)
                train_params['workers'] = 2
                results = model.train(**train_params)
            else:
                raise e
        finally:
            torch.cuda.empty_cache()  # Clean up GPU memory
        
        # Save best model
        weights_dir = Path("runs/detect/taco_yolov8m/weights")
        best_model = next(weights_dir.glob("best*.pt"), None)
        if best_model:
            shutil.copy(best_model, self.model_dir / "taco_best.pt")
            return str(self.model_dir / "taco_best.pt")
        else:
            print("⚠️ No best model found in runs/detect/taco_yolov8m/weights")
            exit(1)

    def run_inference(self, model_path):
        """Run inference on sample images"""
        print("\n🔍 Running inference on test images...")
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"⚠️ Failed to load model: {str(e)}")
            exit(1)
        
        dataset_path = self.data_dir / "taco_yolo_10_class_original"
        test_images = list((dataset_path / "test").glob("*.jpg"))
        
        for img_path in tqdm(test_images[:5], desc="Processing images"):
            try:
                results = model.predict(
                    source=str(img_path),
                    conf=0.25,
                    save=True,
                    save_txt=True,
                    project=str(self.output_dir),
                    device=self.device
                )
                # Plot and close figure to prevent memory leak
                self.plot_results(results[0], img_path.name)
                plt.close('all')  # Close matplotlib figures
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"⚠️ GPU OOM on {img_path.name}, switching to CPU")
                    results = model.predict(
                        source=str(img_path),
                        conf=0.25,
                        save=True,
                        save_txt=True,
                        project=str(self.output_dir),
                        device='cpu'
                    )
                    self.plot_results(results[0], img_path.name)
                    plt.close('all')
                else:
                    print(f"⚠️ Error processing {img_path.name}: {str(e)}")
            finally:
                torch.cuda.empty_cache()  # Clean up GPU memory
    
    def plot_results(self, result, img_name):
        """Visualize detection results with class labels"""
        img = result.orig_img
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            label = f"{result.names[cls_id]} {conf:.2f}"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), font, 0.9, (0, 255, 0), 2)
        
        output_path = str(self.output_dir / img_name)
        cv2.imwrite(output_path, img)
        # Non-blocking plot
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Detections: {img_name}")
        plt.axis('off')
        plt.savefig(output_path.replace('.jpg', '_plot.png'), bbox_inches='tight')
        plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    print("="*50)
    print("TRASH DETECTION WITH YOLOv8 AND TACO DATASET")
    print("="*50)
    
    trainer = TACOTrainer()
    
    # Dataset preparation
    data_yaml = trainer.prepare_dataset()
    
    # Display sample images
    trainer.display_sample_images(num_images=1)
    
    # Model training
    trained_model = trainer.train_model(data_yaml)
    print(f"\n✅ Training complete! Best model saved at: {trained_model}")
    
    # Inference demonstration
    trainer.run_inference(trained_model)
    print(f"\n🎉 Inference results saved to: {trainer.output_dir}")