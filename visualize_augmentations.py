import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from horizon_dataset import HorizonDataset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import random
random.seed(42)  # For reproducible results
torch.manual_seed(42)

def denormalize_image(img_tensor):
    """Convert normalized tensor back to PIL Image for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize
    img_tensor = img_tensor * std + mean
    
    # Clamp values to valid range
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    # Convert to PIL Image
    img_tensor = img_tensor.permute(1, 2, 0).numpy() * 255
    return Image.fromarray(img_tensor.astype(np.uint8))

def draw_horizon_line(img, horizon_y, roll_angle, color=(255, 0, 0), width=2):
    """Draw a horizon line on the image based on Y position and roll angle
    
    Args:
        img: PIL Image to draw on
        horizon_y: Y coordinate in centered coordinate system (-1 to 1, 0 is middle)
        roll_angle: roll angle in degrees (0 is level, positive is clockwise roll)
        color: RGB color for the horizon line
        width: width of the line to draw
    """
    img_width, img_height = img.size
    
    # Convert centered horizon_y to pixel coordinates
    # In centered system, 0 is middle, +1 is top, -1 is bottom
    center_y = img_height // 2
    y_offset = int(-horizon_y * 280 / 2)  # Negate because positive Y is up
    y_pos = center_y + y_offset
    
    # Calculate center point
    center_x = img_width // 2
    
    # Calculate endpoints of the horizon line based on roll angle
    # For small angles, we can use a simplified approach
    if abs(roll_angle) < 1.0:
        # Almost horizontal line
        x1, y1 = 0, y_pos
        x2, y2 = img_width, y_pos
    else:
        # Convert angle to radians
        angle_rad = np.radians(roll_angle)
        # Positive roll = negative slope
        slope = -np.tan(angle_rad)
        
        # Calculate how far to extend the line to reach edges
        # This depends on the angle - steeper angles need shorter lines
        dx = img_width // 2
        if abs(slope) > 1.0:
            dx = int(img_height // (2 * abs(slope)))
        
        # Calculate endpoints
        x1 = center_x - dx
        y1 = int(y_pos - slope * dx)
        
        x2 = center_x + dx
        y2 = int(y_pos + slope * dx)
        
        # Ensure the line extends to image boundaries
        if x1 < 0:
            # Adjust y based on slope
            y1 = int(y1 - slope * x1)
            x1 = 0
        if x2 >= img_width:
            # Adjust y based on slope
            y2 = int(y2 - slope * (x2 - img_width + 1))
            x2 = img_width - 1
            
        if y1 < 0:
            x1 = int(x1 - y1 / slope) if slope != 0 else x1
            y1 = 0
        if y2 >= img_height:
            x2 = int(x2 - (y2 - img_height + 1) / slope) if slope != 0 else x2
            y2 = img_height - 1
    
    # Draw the line
    print(f'drawing line from {x1}, {y1} to {x2}, {y2}')
    draw = ImageDraw.Draw(img)
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    
    # Draw a dot at the center of the image for reference
    center_size = 3
    draw.ellipse([(center_x - center_size, center_y - center_size), 
                 (center_x + center_size, center_y + center_size)], fill=(0, 255, 0))
    
    # Add text with angle information
    text = f"y={horizon_y:.2f}, angle={roll_angle:.1f}°"
    draw.text((10, 10), text, fill=(255, 0, 0))
    
    return img

def main():
    # Path to your dataset
    csv_file = "horizon_data.csv"  # Update this to your CSV file path
    img_dir = "images"              # Update this to your image directory
    
    # Create a dataset with augmentations
    augmented_dataset = HorizonDataset(csv_file, img_dir, train_mode=True)
    original_dataset = HorizonDataset(csv_file, img_dir, train_mode=False)
    
    # Number of samples to visualize
    num_samples = 5
    
    # Create output directory if it doesn't exist
    output_dir = "augmentation_checks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample indices
    indices = torch.randperm(min(len(augmented_dataset), 100))[:num_samples].tolist()
    
    # Create a figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    # Check if we have only one sample
    if num_samples == 1:
        axes = np.array([axes])
    
    for i, idx in enumerate(indices):
        # Get original filename for reference
        img_name = original_dataset.horizon_data.iloc[idx, 0]
        print(f"Processing image {i+1}/{num_samples}: {img_name}")
        
        # Get original sample
        orig_img, orig_targets = original_dataset[idx]
        
        # Get augmented sample (different every time due to random transforms)
        aug_img, aug_targets = augmented_dataset[idx]
        
        # Denormalize both images for visualization
        orig_pil = denormalize_image(orig_img)
        aug_pil = denormalize_image(aug_img)
        
        # Get denormalized target values
        orig_y = orig_targets[0].item()# * original_dataset.avg_y_std + original_dataset.avg_y_mean
        orig_roll = orig_targets[1].item()# * original_dataset.roll_std + original_dataset.roll_mean
        
        aug_y = aug_targets[0].item()# * augmented_dataset.avg_y_std + augmented_dataset.avg_y_mean
        aug_roll = aug_targets[1].item()# * augmented_dataset.roll_std + augmented_dataset.roll_mean
        
        print(f"  Original: y={orig_y:.3f}, roll={orig_roll:.2f}°")
        print(f"  Augmented: y={aug_y:.3f}, roll={aug_roll:.2f}°")
        
        # Draw horizon lines
        orig_pil = draw_horizon_line(orig_pil, orig_y, orig_roll)
        aug_pil = draw_horizon_line(aug_pil, aug_y, aug_roll)
        
        # Display images
        axes[i, 0].imshow(np.array(orig_pil))
        axes[i, 0].set_title(f"Original: y={orig_y:.2f}, roll={orig_roll:.1f}°")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(np.array(aug_pil))
        axes[i, 1].set_title(f"Augmented: y={aug_y:.2f}, roll={aug_roll:.1f}°")
        axes[i, 1].axis('off')
        
        # Save individual paired images
        paired = Image.new('RGB', (orig_pil.width + aug_pil.width, orig_pil.height))
        paired.paste(orig_pil, (0, 0))
        paired.paste(aug_pil, (orig_pil.width, 0))
        paired.save(os.path.join(output_dir, f"sample_{i}_{os.path.basename(img_name)}"))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "augmentation_visualization.png"), dpi=150)
    
    print(f"\nVisualization saved to {output_dir}/")
    print("Individual samples and combined visualization have been saved.")

if __name__ == "__main__":
    main()