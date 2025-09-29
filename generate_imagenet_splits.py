import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv

def get_imagenet_train_images(data_root):
    """Extract training images and their labels from the train directory"""
    ilsvrc_path = Path(data_root) / "ILSVRC"
    train_dir = ilsvrc_path / "Data" / "CLS-LOC" / "train"
    train_data = []

    if not train_dir.exists():
        raise FileNotFoundError(f"Cannot find training directory. Checked: {train_dir}.")

    print(f"Processing training images from: {train_dir}")

    # Each subdirectory represents a synset (class)
    for synset_dir in train_dir.iterdir():
        synset_id = synset_dir.name
        
        # Get all images in this synset directory
        for img_file in synset_dir.iterdir():
            train_data.append({
                "image_path": str(img_file.relative_to(data_root)),
                "synset_id": synset_id,
            })

    print(f"Found {len(train_data)} training images across {len(set(item['synset_id'] for item in train_data))} classes")
    return train_data

def get_imagenet_val_images(data_root):
    """Extract validation images and their labels"""
    ilsvrc_path = Path(data_root) / "ILSVRC"
    val_dir = ilsvrc_path / "Data" / "CLS-LOC" / "val"
    val_solution_file = Path(data_root) / "LOC_val_solution.csv"
    
    if not val_dir.exists():
        raise FileNotFoundError(f"Cannot find validation directory. Checked: {val_dir}.")
    if not val_solution_file.exists():
        raise FileNotFoundError(f"Cannot find validation solution file. Checked: {val_solution_file}.")

    val_data = []
    
    # Load validation labels
    val_labels = {}
    with open(val_solution_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["ImageId"]
            # Extract synset from prediction string (first element)
            pred_parts = row["PredictionString"].strip().split()
            synset_id = pred_parts[0]
            val_labels[img_id] = synset_id
    
    print(f"Processing validation images from: {val_dir}")
    
    # Get all validation images
    for img_file in val_dir.iterdir():
        img_id = img_file.stem  # filename without extension

        # Get synset and label
        synset_id = val_labels.get(img_id, "unknown")
        if synset_id == "unknown":
            print(f"Warning: No label found for validation image {img_id}")

        val_data.append({
            "image_path": str(img_file.relative_to(data_root)),
            "synset_id": synset_id,
        })
    
    print(f"Found {len(val_data)} validation images")
    return val_data

def create_splits(data_root):
    """Create train, validation, and test splits"""
    print("Creating dataset splits...")

    # Get original training and validation data
    train_val_data = get_imagenet_train_images(data_root)
    test_data = get_imagenet_val_images(data_root)
    
    # Split train and val
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=50000,
        random_state=42,
        shuffle=True
    )
        
    # Create final dataset splits
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    # Print split statistics
    print(f"\nDataset split summary:")
    print(f"Training samples: {len(splits['train'])}")
    print(f"Validation samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")
        
    return splits

def main():
    # Configuration
    DATA_ROOT = "/datasets/imagenet"  # Update this path
    OUTPUT_FILE = "/datasets/imagenet/splits.pkl"
    
    # Check if data root exists
    if not Path(DATA_ROOT).exists():
        print(f"Error: Data root '{DATA_ROOT}' does not exist.")
        print("Please update DATA_ROOT to point to your ImageNet dataset directory.")
        return

    splits = create_splits(DATA_ROOT)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(splits, f)

    print(f"\nSuccessfully created dataset splits!")
    print(f"Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
