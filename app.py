import torch
import timm
from torch import nn
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms
from io import BytesIO
from datetime import datetime

import os
import requests

# Initialize Flask app
app = Flask(__name__)

# === Download the model from Google Drive ===
def download_model_from_gdrive():
    url = "https://drive.google.com/uc?export=download&id=13ZHfqvOepNf4wHMaTd_EUxhbPyuxhhmo"
    model_path = "vit_sanitization.pth"
    if not os.path.exists(model_path):  # Avoid re-downloading
        print("Downloading model from Google Drive...")
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")

# === Load model and handle 'module.' prefix ===
def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove "module." prefix
    model.load_state_dict(state_dict, strict=False)
    return model

# === Download model weights before anything else ===
download_model_from_gdrive()

# === Build and load model ===
model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.head = nn.Linear(model.head.in_features, 2)
model = load_model(model, 'vit_sanitization.pth')
model.eval()

# === Class names for output ===
CLASS_NAMES = ['Bad', 'Good']

# === Image transformation ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Prediction helper ===
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(torch.device('cpu'))
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return CLASS_NAMES[predicted.item()]

# === Flask route ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(BytesIO(file.read()))
        img_path = 'temp_image.jpg'
        img.save(img_path)

        predicted_class = predict_image(img_path)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result_image_path = f'predicted_{predicted_class}_{timestamp}.jpg'
        img.save(result_image_path)

        return jsonify({'prediction': predicted_class, 'saved_image': result_image_path})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run the app ===
if __name__ == '__main__':
    app.run(debug=True)
