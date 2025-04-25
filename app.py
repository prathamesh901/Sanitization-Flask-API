import torch
import timm
from torch import nn
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms
from io import BytesIO
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Function to load the model and handle potential 'module.' prefix
def load_model(model, model_path):
    # Load the saved state_dict
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Remove the 'module.' prefix if present
    state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove "module." prefix
    
    # Load the state_dict into the model
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore missing keys
    return model

# Load the trained model
model = timm.create_model('vit_base_patch16_224', pretrained=False)  # Load model without pre-trained weights

# Adjust the output layer to have 2 classes (for your specific task)
model.head = nn.Linear(model.head.in_features, 2)  # Assuming 2 classes for your case

# Load your saved model weights
model = load_model(model, 'vit_sanitization.pth')
model.eval()

# Define the class names (Modify as per your dataset)
CLASS_NAMES = ['Bad', 'Good']  # Replace with your class labels

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Define a function to predict the class of an image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(torch.device('cpu'))  # Ensure the image tensor is on CPU
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return CLASS_NAMES[predicted.item()]

# API route to upload and predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded image temporarily
        img = Image.open(BytesIO(file.read()))
        img_path = 'temp_image.jpg'
        img.save(img_path)

        # Predict the image class
        predicted_class = predict_image(img_path)
        
        # Get the current timestamp to include in the filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result_image_path = f'predicted_{predicted_class}_{timestamp}.jpg'
        
        # Save the image with the new name
        img.save(result_image_path)

        # Return the prediction result and the saved image path
        return jsonify({'prediction': predicted_class, 'saved_image': result_image_path})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
