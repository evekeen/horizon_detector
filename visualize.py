import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from horizon_model import HorizonNetLight
import math

def load_model(model_path, device):
    model = HorizonNetLight(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_horizon(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Denormalize predictions
    avg_y = output[0, 0].item() * 5000.0
    roll_angle = output[0, 1].item() * 90.0
    
    print(f"Predicted avg_y: {avg_y:.2f}, roll_angle: {roll_angle:.2f} degrees")
    
    # Calculate the horizon line endpoints
    width, height = original_size
    
    # Convert from MATLAB coordinate system to pixel coordinates
    # In MATLAB: origin is at center, y-axis points down
    # In pixel coordinates: origin is at top-left, y-axis points down
    
    # Calculate the center of the image in MATLAB coordinates
    center_x = (width + 1) / 2 - 1
    center_y = (height + 1) / 2 - 1
    
    # Convert avg_y from MATLAB to pixel coordinates
    pixel_avg_y = center_y - avg_y
    
    # Calculate endpoints of the horizon line
    # Use the full width of the image
    x_left = 0
    x_right = width - 1
    
    # Calculate the y-coordinates based on the roll angle
    # tan(roll_angle) = (y_right - y_left) / (x_right - x_left)
    # We need to adjust the sign because of the different coordinate systems
    slope = math.tan(math.radians(roll_angle))
    
    # Calculate the y-coordinates
    half_width = (x_right - x_left) / 2
    y_center = pixel_avg_y
    y_left = y_center - slope * half_width
    y_right = y_center + slope * half_width
    
    return (x_left, y_left), (x_right, y_right)

def visualize_horizon(image_path, left_point, right_point):
    image = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(np.array(image))
    
    # Plot the horizon line
    plt.plot([left_point[0], right_point[0]], [left_point[1], right_point[1]], 'g-', linewidth=3)
    
    plt.title('Predicted Horizon Line')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MPS not available, using {device}")
    
    model_path = 'best_horizon_model.pth'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Load the model
    model = load_model(model_path, device)
    print("Model loaded successfully")
    
    # Get a list of test images
    image_dir = 'images'
    image_files = os.listdir(image_dir)
    image_files = [f for f in image_files if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    # Visualize a few random images
    import random
    random.seed(42)
    test_images = random.sample(image_files, min(5, len(image_files)))
    
    for image_file in test_images:
        image_path = os.path.join(image_dir, image_file)
        print(f"\nProcessing {image_path}")
        
        left_point, right_point = predict_horizon(model, image_path, device)
        visualize_horizon(image_path, left_point, right_point)
        
        input("Press Enter to continue...")

if __name__ == "__main__":
    main()
