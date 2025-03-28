import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class HorizonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.horizon_data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
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
        
        image = self.transform(image)
        
        avg_y = self.horizon_data.iloc[idx, 1]
        roll_angle = self.horizon_data.iloc[idx, 2]
        
        # Roll angle is already in degrees [-90, 90]
        # Normalize to [-1, 1]
        roll_angle_normalized = roll_angle / 90.0
        
        targets = torch.tensor([avg_y, roll_angle_normalized], dtype=torch.float)
        
        return image, targets


def create_data_loaders(csv_file, img_dir, batch_size=32, train_split=0.8, val_split=0.1):
    dataset = HorizonDataset(csv_file, img_dir)
    
    # Split the dataset into train, validation, and test sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_end = int(train_split * dataset_size)
    val_end = train_end + int(val_split * dataset_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, val_loader, test_loader
