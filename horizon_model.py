import torch
import torch.nn as nn
import torchvision.models as models

class HorizonNet(nn.Module):
    def __init__(self, pretrained=True):
        super(HorizonNet, self).__init__()
        
        resnet = models.resnet50(weights='DEFAULT' if pretrained else None)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.regression_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Output: [avg_y, roll_angle]
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.regression_head(x)
        return x


class HorizonNetLight(nn.Module):
    def __init__(self, pretrained=True):
        super(HorizonNetLight, self).__init__()
        
        # Use EfficientNet backbone which has better performance with similar cost
        self.backbone = models.efficientnet_b0(weights='DEFAULT' if pretrained else None).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Number of features from EfficientNet-B0
        backbone_features = 1280
        
        # More sophisticated regression head with residual connections
        self.fc1 = nn.Linear(backbone_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Separate heads for y-coordinate and roll angle for better specialization
        self.avg_y_head = nn.Linear(128, 1)
        self.roll_head = nn.Linear(128, 1)
        
    def forward(self, x):
        # Extract features
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        # First layer with residual connection
        identity = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, p=0.3, training=self.training)
        
        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, p=0.3, training=self.training)
        
        # Third layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        
        # Separate prediction heads
        avg_y = self.avg_y_head(x)
        roll = self.roll_head(x)
        
        # Combine outputs
        return torch.cat([avg_y, roll], dim=1)
