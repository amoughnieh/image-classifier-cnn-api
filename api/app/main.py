import os
import torch
import torch.nn as nn
import torch.nn.functional as func
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import numpy as np
from typing import List

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

class SimpleCNN(nn.Module):
    def __init__(self, seed=42):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (224 // 4) * (224 // 4), 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the pre-trained model
model = SimpleCNN()
model.load_state_dict(torch.load('best_model/CNN_best_model-f1_0.7887.pth'))
model.eval()

app = FastAPI(title="Image Classifier API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_valid_file_extension(filename: str) -> bool:
    """Check if the file extension is allowed"""
    return filename.lower().endswith(tuple(f'.{ext}' for ext in ALLOWED_EXTENSIONS))

def is_valid_file_size(file_size: int) -> bool:
    """Check if the file size is within limits"""
    return file_size <= MAX_FILE_SIZE

def prepare_image_for_inference(image_data):
    """Prepare image from bytes data"""
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_data))

        # Convert image to RGB
        img = img.convert('RGB').resize((224, 224))
        img = np.array(img).transpose(2, 0, 1)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        # Normalize image
        img_tensor = img_tensor / 255.0
        mean_tor = img_tensor.mean(dim=(0, 2, 3), keepdim=True)
        std_tor = img_tensor.std(dim=(0, 2, 3), keepdim=True)
        img_tensor = (img_tensor - mean_tor) / std_tor
        return img_tensor
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error processing image: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image
    """
    # Validate file extension
    if not is_valid_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read image file
    image_data = await file.read()

    # Validate file size
    if not is_valid_file_size(len(image_data)):
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
        )

    try:
        # Prepare image
        img_tensor = prepare_image_for_inference(image_data)

        # Perform inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # Get predicted class and probabilities
        label_dict = {0: 'Car', 1: 'Plane', 2: 'Ship'}
        pred_class = np.argmax(probabilities)
        predicted_values = {label_dict[key]: round(val, 3)
                            for key, val in enumerate(probabilities.tolist())}

        result = {
            "Predicted Image Class": label_dict[pred_class],
            "Certainty": f"{probabilities[pred_class]:.2%}",
            "Probabilities": predicted_values,
        }

        return {
            'Predicted Class': result["Predicted Image Class"],
            'Certainty': result["Certainty"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)