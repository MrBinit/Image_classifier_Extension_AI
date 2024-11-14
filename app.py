from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image, UnidentifiedImageError
import requests
import os
import base64

app = Flask(__name__)
CORS(app)

# Define model loading directory and download directory
load_directory = "/home/binit/image_classification/models"
download_directory = "/home/binit/image_classification/image_download"
os.makedirs(download_directory, exist_ok=True)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained(load_directory)
model = AutoModelForImageClassification.from_pretrained(load_directory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class ImageURL(BaseModel):
    url: str

def download_image(image_url):
    # Ensure the URL uses HTTPS
    if image_url.startswith("http://"):
        image_url = image_url.replace("http://", "https://", 1)

    if image_url.startswith('data:image'):
        return save_base64_image(image_url)
    
    response = requests.get(image_url)
    if response.status_code == 200:
        image_path = 'downloaded.jpg'  
        with open(image_path, 'wb') as file:
            file.write(response.content)
        return image_path
    else:
        raise Exception('Failed to download image')

def save_base64_image(base64_string):
    header, encoded = base64_string.split(',', 1)
    if 'jpeg' in header:
        ext = 'jpg'
    elif 'png' in header:
        ext = 'png'
    else:
        raise Exception('Unsupported image format')

    image_data = base64.b64decode(encoded)
    image_path = f'downloaded.{ext}'
    
    with open(image_path, 'wb') as file:
        file.write(image_data)
    
    return image_path

def delete_image(image_path):
    if os.path.exists(image_path):
        os.remove(image_path)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'message':"active"})

@app.route('/upload-image', methods=['POST'])
def classify_image():
    # Download or decode the image
    data = request.json
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    try:
        image_path = download_image(image_url)
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        delete_image(image_path)
        return jsonify({'error': 'Could not identify the downloaded image.'}), 500
    except Exception as e:
        delete_image(image_path)
        return jsonify({'error': str(e)}), 500

    try:
        # Process image for model prediction
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get predicted label
        predicted_class_idx = logits.argmax(-1).item()
        id2label = model.config.id2label
        predicted_label = id2label[predicted_class_idx]

# Check if the label is 'artificial' and set `is_ai_image` accordingly
        is_ai_image = True if predicted_label == 'human' else False

# Return JSON response with the boolean value
        return jsonify({"is_ai_image": is_ai_image})

    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        delete_image(image_path)  # Clean up the downloaded image file

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
