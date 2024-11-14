import os
import yaml
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import numpy as np

# Load configuration from config.yaml
with open("/home/binit/image_classification/config.yaml", "r") as file:
    config = yaml.safe_load(file)
train_dir = config["paths"]["train_dir"]
test_dir = config["paths"]["test_dir"]
val_dir = config["paths"]["val_dir"]
pretrained_model = config["models"]["model_name"]
image_extensions = tuple(config["models"]["image_extensions"])
print("Loaded paths and model successfully")

# Initialize the image processor and model
image_processor = AutoImageProcessor.from_pretrained(pretrained_model)
model = AutoModelForImageClassification.from_pretrained(pretrained_model)

# Function to process and save images
def process_and_save_images(source_path):
    print(f"Processing images from '{source_path}'")

    if not os.path.exists(source_path):
        print(f"Source folder '{source_path}' doesn't exist. Skipping.")
        return
    
    for filename in os.listdir(source_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(source_path, filename)
            save_path = os.path.join(source_path, f"{filename}.pt")

            # Skip if already processed
            if os.path.exists(save_path):
                print(f"Skipping already processed image: {filename}")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # Convert to [channels, height, width]
                
                torch.save(image_tensor, save_path)
                print(f"Processed and saved: {save_path}")

                # delete original file after processing
                os.remove(image_path)
                print(f"Deleted original image: {image_path}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        else:
            print(f"Skipped non-image file: {filename}")


# Process each data split: train, test, and validation
split_paths = {
    "train": train_dir,
    "test": test_dir,
    "validation": val_dir
}

for split_name, split_dir in split_paths.items():
    for folder_name in ["real", "AI"]:
        source_folder_path = os.path.join(split_dir, folder_name)
        process_and_save_images(source_folder_path)

print("All processing completed.")
