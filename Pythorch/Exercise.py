import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class RandomMRIDataset(Dataset):
    def __init__(self, length=100):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.randn(1, 64, 64)      # fake MRI slice
        mask = (torch.randn(1, 64, 64) > 0).float()  # fake mask
        return image, mask

dataset = RandomMRIDataset()
img, msk = dataset[0]
img.shape, msk.shape

import torch.nn.functional as F

class RandomMRIDataset(Dataset):
    def __init__(self, length=100):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate 64x64
        image = torch.randn(1, 64, 64)
        mask = (torch.randn(1, 64, 64) > 0).float()
        
        # Resize to 128x128 using bilinear interpolation
        # unsqueeze(0) adds a batch dimension: [1, 1, 64, 64]
        image = F.interpolate(image.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)
        
        return image, mask
    
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Define the layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 1),
            nn.Sigmoid()
        )

    # 2. Define the data flow (MANDATORY)
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU()
    )

class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = conv_block(1, 16)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv_block(16, 32)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = conv_block(32 + 16, 16) # +16 due to skip connection
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x2 = self.enc2(x2)
        
        # Decoder with Skip Connection
        out = self.up(x2)
        out = torch.cat([out, x1], dim=1) # Concatenate along channel dim
        out = self.dec1(out)
        return self.final(out)
    
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs = torch.sigmoid(inputs) # Ensure inputs are between 0 and 1
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
loss_history = []

# 1. Initialize the custom dataset
dataset = RandomMRIDataset() 

# 2. Create the DataLoader (this creates the 'loader' variable)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for epoch in range(5):
    epoch_loss = 0
    for images, _ in loader: # Using dummy targets for simplicity
        optimizer.zero_grad()
        preds = model(images)
        
        target = torch.zeros_like(preds) # Fake target
        loss = criterion(preds, target)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")

# Plotting
plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
