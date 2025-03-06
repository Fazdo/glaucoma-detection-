import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

class ImageAnalyzer:
    def __init__(self, model_path='./models/glaucoma_model.pth'):
        # Determine device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create MobileNetV3Large model for binary classification
        self.model = models.mobilenet_v3_large(pretrained=False)
        self.model.classifier[-1] = torch.nn.Linear(1280, 2)  # 2 classes: with/without glaucoma
        
        # Load model weights if available
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model successfully loaded from {model_path}")
                self.model.eval()  # Switch to evaluation mode
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using model without pre-training")
        else:
            print("Model path not specified or file doesn't exist. Using model without pre-training")
        
        # Move model to appropriate device
        self.model = self.model.to(self.device)
        
        # Define transformations for image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.class_names = ['No Glaucoma', 'Glaucoma']

    def preprocess_image(self, image):
        """Preprocess image for MobileNetV3 model"""
        # Apply transformations
        img_tensor = self.transform(image)
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        # Move tensor to appropriate device
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def analyze(self, image):
        """Analyze image and return prediction results"""
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Perform prediction
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get index of class with maximum probability
        _, predicted_class = torch.max(outputs, 1)
        class_idx = predicted_class.item()
        
        # Convert probability tensor to numpy array
        probabilities = probabilities.cpu().numpy()
        
        # Format result
        result = {
            "class": self.class_names[class_idx],
            "confidence": float(probabilities[class_idx]),
            "probabilities": {self.class_names[i]: float(probabilities[i]) for i in range(len(self.class_names))},
            "description": "Glaucoma signs detected" if class_idx == 1 else "No glaucoma signs detected"
        }
        
        return result
