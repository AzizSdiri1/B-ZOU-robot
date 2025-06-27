import yaml
from pathlib import Path
from collections import Counter

def create_yaml(dataset_path):
    """Auto-generate data.yaml for TACO 10-class dataset"""
    # Define TACO class names (standard 10 classes)
    class_names = [
        "bottle",
        "bottle cap",
        "can",
        "battery" ,
        "cup",
        " blister pack",
        "Carton",
        "aluminum foil",
        "Lid",
        "Other"
    ]
    
    # Verify class distribution in annotations
    class_counts = Counter()
    for split in ['train', 'valid', 'test']:
        txt_files = list(Path(dataset_path).glob(f"{split}/*.txt"))
        for txt_file in txt_files:
            with open(txt_file) as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
    
    print("\n📊 Dataset Class Distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  Class {class_id}: {class_names[class_id]} - {count} instances")
    
    # Create YAML content
    yaml_content = {
        'path': str(Path(dataset_path).resolve()),
        'train': "train",
        'val': "valid",
        'test': "test",
        'names': class_names,
        'nc': len(class_names)
    }
    
    # Save to file
    yaml_path = Path(dataset_path) / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"\n✅ Created YAML at: {yaml_path}")
    return yaml_path

# Example usage:
dataset_path = "taco_yolo_project/data/taco_yolo_10_class_original"
yaml_file = create_yaml(dataset_path)