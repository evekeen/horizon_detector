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
        
        mobilenet = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
        
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.regression_head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # Output: [avg_y, roll_angle]
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.regression_head(x)
        return x
