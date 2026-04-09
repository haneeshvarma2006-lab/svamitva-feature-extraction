# SVAMITVA Feature Extraction

AI-based geospatial feature extraction from drone orthophotos using PyTorch.
Built for the MoPR Geospatial Intelligence Challenge at IIT Tirupati.

## Team - The Alethians
SVCE Tirupati

## Problem Statement
SVAMITVA drone orthophotos contain rich geospatial data but manual
feature extraction is slow and error-prone. This project automates
extraction of land features using deep learning.

## Approach
- Tiled large drone orthophotos into 720x720 patches
- Built a custom CNN Encoder-Decoder model in PyTorch
- Trained on 352 labeled image-mask pairs
- Model segments features directly from aerial imagery

## Features Detected
Buildings, Roads, Water Bodies, Utilities

## Model Details
- Architecture: Custom CNN Encoder-Decoder
- Loss Function: BCELoss
- Optimizer: Adam (lr=0.001)
- Epochs: 30
- Final Loss: 15.3

## File Structure
- model.py — CNN architecture
- dataset_loader.py — Data loading pipeline
- predict.py — Training and inference script
- village_model.pth — Trained model weights

## Installation
pip install torch torchvision opencv-python-headless numpy pillow

## Usage
python predict.py

## GitHub
github.com/haneeshvarma2006-lab/svamitva-feature-extraction
