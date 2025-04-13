import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
import logging
from fabric_classifier import FabricClassifier, analyze_color, analyze_texture

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path, num_classes):
    """Load the trained model"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FabricClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path, transform):
    """Preprocess image for prediction"""
    try:
        image = Image.open(image_path).convert('RGB')
        # Save original image for analysis
        original_image = image.copy()
        # Apply transformations
        transformed_image = transform(image)
        return transformed_image, original_image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise

def predict_fabric(model, image, device, class_names):
    """Make prediction on a single image"""
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = probabilities.topk(5, dim=1)
        
        predictions = []
        for i in range(5):
            predictions.append({
                'class': class_names[top_class[0][i].item()],
                'probability': top_prob[0][i].item() * 100
            })
    
    return predictions

def analyze_fabric(image):
    """Analyze fabric properties"""
    # Color analysis
    hist_h, hist_s, hist_v = analyze_color(image)
    
    # Texture analysis
    glcm = analyze_texture(image)
    
    # Calculate color distribution
    color_distribution = {
        'hue': np.mean(hist_h),
        'saturation': np.mean(hist_s),
        'value': np.mean(hist_v)
    }
    
    # Calculate texture properties
    texture_properties = {
        'contrast': np.mean(glcm),
        'homogeneity': np.std(glcm)
    }
    
    return color_distribution, texture_properties

def main():
    try:
        # Define class names (must match training)
        class_names = [
            'Acrylic', 'Artificial_fur', 'Artificial_leather', 'Blended', 'Chenille',
            'Corduroy', 'Cotton', 'Crepe', 'Denim', 'Felt', 'Fleece', 'Fur',
            'Leather', 'Linen', 'Lut', 'Nylon', 'Polyester', 'Satin', 'Silk',
            'Suede', 'Terrycloth', 'Unclassified', 'Utilities', 'Velvet',
            'Viscose', 'Wool'
        ]
        
        # Load model
        model_path = 'best_fabric_classifier.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
            
        logger.info("Loading model...")
        model, device = load_model(model_path, len(class_names))
        
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get image path from user
        image_path = input("Enter the path to the fabric image: ")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        
        # Preprocess image
        logger.info("Preprocessing image...")
        transformed_image, original_image = preprocess_image(image_path, transform)
        
        # Make prediction
        logger.info("Making prediction...")
        predictions = predict_fabric(model, transformed_image, device, class_names)
        
        # Analyze fabric properties
        logger.info("Analyzing fabric properties...")
        color_dist, texture_props = analyze_fabric(original_image)
        
        # Print results
        print("\n=== Fabric Classification Results ===")
        print("\nTop 5 Predictions:")
        for pred in predictions:
            print(f"{pred['class']}: {pred['probability']:.2f}%")
        
        print("\n=== Fabric Analysis ===")
        print("\nColor Properties:")
        print(f"Hue: {color_dist['hue']:.2f}")
        print(f"Saturation: {color_dist['saturation']:.2f}")
        print(f"Value: {color_dist['value']:.2f}")
        
        print("\nTexture Properties:")
        print(f"Contrast: {texture_props['contrast']:.2f}")
        print(f"Homogeneity: {texture_props['homogeneity']:.2f}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    main() 