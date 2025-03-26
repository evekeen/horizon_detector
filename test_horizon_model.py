import unittest
import torch
import os
import numpy as np
from horizon_model import HorizonNetLight, HorizonNet
from horizon_dataset import HorizonDataset

class TestHorizonModel(unittest.TestCase):
    def setUp(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_light = HorizonNetLight(pretrained=False)
        self.model_full = HorizonNet(pretrained=False)
    
    def test_model_forward_light(self):
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        self.model_light.eval()
        with torch.no_grad():
            output = self.model_light(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, 2))
        
    def test_model_forward_full(self):
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        self.model_full.eval()
        with torch.no_grad():
            output = self.model_full(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, 2))
    
    def test_model_to_device(self):
        self.model_light.to(self.device)
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        self.model_light.eval()
        with torch.no_grad():
            output = self.model_light(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, 2))
        self.assertEqual(output.device.type, self.device.type)
    
    def test_dataset_processing(self):
        if os.path.exists('horizon_data.csv'):
            dataset = HorizonDataset('horizon_data.csv', 'images')
            self.assertGreater(len(dataset), 0)
            
            image, targets = dataset[0]
            self.assertEqual(image.shape, (3, 224, 224))
            self.assertEqual(targets.shape, (2,))
            
            # Check if targets are normalized
            self.assertTrue(-1.0 <= targets[0] <= 1.0)
            self.assertTrue(-1.0 <= targets[1] <= 1.0)

if __name__ == '__main__':
    unittest.main()
