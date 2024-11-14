import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
model = AutoModelForImageClassification.from_pretrained("/home/binit/image_classification/checkpoint-28600")

# Load and preprocess the image
image_path = "/home/binit/image_classification/images.jpeg"
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
