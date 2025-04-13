import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from collections import defaultdict
import logging
from tqdm import tqdm
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Checkpoint directory
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(epoch, model, optimizer, best_val_acc, train_loss, val_loss, train_acc, val_acc, filename='checkpoint.pth'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, filename))
    logger.info(f"Checkpoint saved to {os.path.join(CHECKPOINT_DIR, filename)}")

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    """Load training checkpoint"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['best_val_acc']
    return 0, 0.0

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

class FabricDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Check if root directory exists
        if not os.path.exists(root_dir):
            raise ValueError(f"Root directory {root_dir} does not exist")
        
        # Get list of classes (subdirectories)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
        
        if not self.classes:
            raise ValueError(f"No valid class directories found in {root_dir}")
            
        logger.info(f"Found {len(self.classes)} classes: {self.classes}")
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        
        # Count total images
        total_images = 0
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_images = []
            
            # Recursively search for images in all subdirectories
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')) and not file.startswith('.'):
                        class_images.append(os.path.join(root, file))
            
            total_images += len(class_images)
            logger.info(f"Found {len(class_images)} images in class {class_name}")
            
            for img_path in class_images:
                self.samples.append((img_path, self.class_to_idx[class_name]))
        
        if not self.samples:
            raise ValueError(f"No valid images found in any class directory under {root_dir}")
            
        logger.info(f"Total images loaded: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise

class FabricClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FabricClassifier, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def analyze_color(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    
    # Calculate color histogram
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    return hist_h, hist_s, hist_v

def analyze_texture(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Calculate texture features using GLCM
    glcm = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return glcm

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, resume=False):
    device = get_device()
    model = model.to(device)
    
    start_epoch = 0
    best_val_acc = 0.0
    
    if resume:
        start_epoch, best_val_acc = load_checkpoint(model, optimizer)
        logger.info(f"Resuming training from epoch {start_epoch + 1}")
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Log epoch summary
        logger.info(f'Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint after each epoch
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_acc=best_val_acc,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_fabric_classifier.pth')
            logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')

def main():
    try:
        # Data preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = FabricDataset('Fabrics', transform=transform)
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty after loading")
            
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        logger.info(f"Training set size: {len(train_dataset)}")
        logger.info(f"Validation set size: {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = FabricClassifier(num_classes=len(dataset.classes))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model (set resume=True to continue from last checkpoint)
        train_model(model, train_loader, val_loader, criterion, optimizer, resume=True)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 