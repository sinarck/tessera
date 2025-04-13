# Fabric Classifier

A deep learning-based fabric classification system that can identify and classify different types of fabrics from images. This project uses PyTorch and computer vision techniques to analyze fabric textures and patterns.

## Features

- Fabric classification using deep learning
- Texture and pattern analysis
- Color analysis
- Pre-trained model support
- Training and inference capabilities
- Checkpoint saving and loading

## Requirements

- Python 3.x
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- Pillow >= 9.0.0
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- scikit-learn >= 1.0.0

## Installation

1. Clone this repository:

```bash
git clone [your-repository-url]
cd tessera
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── Fabrics/              # Directory containing fabric images
├── checkpoints/         # Directory for saving model checkpoints
├── fabric_classifier.py # Main training and model definition
├── predict_fabric.py    # Script for making predictions
├── requirements.txt     # Project dependencies
└── best_fabric_classifier.pth # Pre-trained model weights
```

## Usage

### Training

To train the model:

```bash
python fabric_classifier.py
```

The script will:

- Load and preprocess the fabric images
- Train the model
- Save checkpoints during training
- Evaluate the model on validation data

### Prediction

To make predictions on new fabric images:

```bash
python predict_fabric.py --image path/to/your/image.jpg
```

## Model Architecture

The project uses a custom neural network architecture based on PyTorch, with the following features:

- Convolutional layers for feature extraction
- Fully connected layers for classification
- Support for transfer learning
- Custom data augmentation
