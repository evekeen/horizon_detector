# Horizon Detector

A deep learning system for detecting horizon lines in images using PyTorch.

## Models

- `HorizonNet`: ResNet50-based model for horizon detection
- `HorizonNetLight`: MobileNetV3-small-based lightweight model for horizon detection

Both models predict two values:
- Average Y coordinate of the horizon
- Roll angle of the horizon

## Dataset

The implementation uses the HLW (Horizon Lines in the Wild) dataset:

- `HorizonDataset`: PyTorch dataset class for loading and preprocessing horizon images
- Includes data normalization and augmentation
- Handles train/validation/test splits via `create_data_loaders` function
