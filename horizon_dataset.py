import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class HorizonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, train_mode=False):
        self.horizon_data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.train_mode = train_mode
        
        # Calculate normalization statistics for better handling of the target values
        avg_y_values = self.horizon_data.iloc[:, 1].values
        roll_values = self.horizon_data.iloc[:, 2].values
        
        # Store dataset statistics for normalization
        self.avg_y_mean = np.mean(avg_y_values)
        self.avg_y_std = np.std(avg_y_values)
        self.roll_mean = np.mean(roll_values)
        self.roll_std = np.std(roll_values)
        
        # Standard image normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        
        if transform is None:
            # Base transformation pipeline
            base_transform = [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]
            
            if train_mode:
                # Add data augmentation for training
                self.transform = transforms.Compose([
                    transforms.Resize((280, 280)),
                    transforms.RandomCrop(224),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),  # Small rotations to avoid affecting the roll angle too much
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = transforms.Compose(base_transform)
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.horizon_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.img_dir, self.horizon_data.iloc[idx, 0])
        
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a placeholder black image if the image can't be loaded
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations
        image = self.transform(image)
        
        avg_y = self.horizon_data.iloc[idx, 1]
        roll_angle = self.horizon_data.iloc[idx, 2]
        
        # Z-score normalization for more stable training
        avg_y_normalized = (avg_y - self.avg_y_mean) / (self.avg_y_std + 1e-6)
        roll_angle_normalized = (roll_angle - self.roll_mean) / (self.roll_std + 1e-6)
        
        targets = torch.tensor([avg_y_normalized, roll_angle_normalized], dtype=torch.float)
        
        # Original values for debugging or custom loss functions
        orig_values = torch.tensor([avg_y, roll_angle], dtype=torch.float)
        
        if self.train_mode:
            return image, targets
        else:
            return image, targets, orig_values


def create_data_loaders(csv_file, img_dir, batch_size=32, train_split=0.8, val_split=0.1):
    """
    Create data loaders with improved handling of dataset splits
    
    Args:
        csv_file: Path to CSV file with horizon data
        img_dir: Directory containing images
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
        val_split: Fraction of data to use for validation
    """
    # Create train dataset with augmentations
    train_dataset = HorizonDataset(csv_file, img_dir, train_mode=True)
    
    # Create validation and test datasets without augmentations
    val_dataset = HorizonDataset(csv_file, img_dir, train_mode=False)
    test_dataset = HorizonDataset(csv_file, img_dir, train_mode=False)
    
    
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_end = int(train_split * dataset_size)
    val_end = train_end + int(val_split * dataset_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    
    # Create data loaders with appropriate batch sizes and number of workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, 
                            num_workers=4, pin_memory=True)
    
    # Store dataset statistics for denormalization during evaluation
    train_loader.dataset.avg_y_mean = train_dataset.avg_y_mean
    train_loader.dataset.avg_y_std = train_dataset.avg_y_std
    train_loader.dataset.roll_mean = train_dataset.roll_mean
    train_loader.dataset.roll_std = train_dataset.roll_std
    
    val_loader.dataset.avg_y_mean = train_dataset.avg_y_mean
    val_loader.dataset.avg_y_std = train_dataset.avg_y_std
    val_loader.dataset.roll_mean = train_dataset.roll_mean
    val_loader.dataset.roll_std = train_dataset.roll_std
    
    test_loader.dataset.avg_y_mean = train_dataset.avg_y_mean
    test_loader.dataset.avg_y_std = train_dataset.avg_y_std
    test_loader.dataset.roll_mean = train_dataset.roll_mean
    test_loader.dataset.roll_std = train_dataset.roll_std
    
    return train_loader, val_loader, test_loader