import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from extension_converter import process_single_image  # Import the single image conversion function

# Define the specific image path
image_path = "/home/binit/image_classification/Midjourney-to-ban-Biden-Trump-images-ahead-of-2024-US-elections-fearing-AI-generated-misinformation.avif"

# Check if the image is already in a supported format
supported_formats = ('.jpeg', '.jpg')
convertible_formats = ('.avif', '.png', '.webp')

# Determine if conversion is needed
if image_path.lower().endswith(supported_formats):
    print(f"No conversion needed for {image_path}. Proceeding with classification.")
elif image_path.lower().endswith(convertible_formats):
    print(f"Converting {image_path} to JPEG format...")
    jpeg_image_path = process_single_image(image_path)
    
    # Ensure conversion was successful
    if jpeg_image_path:
        image_path = jpeg_image_path  # Update path to the converted JPEG
    else:
        print(f"Error: Could not convert {image_path}. Exiting.")
        exit(1)
else:
    print(f"Unsupported file format for {image_path}. Exiting.")
    exit(1)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
model = AutoModelForImageClassification.from_pretrained("/home/binit/image_classification/checkpoint-28600")

# Load and preprocess the image
try:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1).item()

    # Interpret the classification result
    if predicted_class_idx == 0:
        print("The image is classified as AI GENERATED.")
    else:
        print("The image is classified as NOT AI GENERATED.")

except Exception as e:
    print(f"Failed to load and classify the image: {e}")
