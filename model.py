# model.py
# nn: Used to define layers (Conv2d, Linear, etc.)
# F: Functional tools for activation functions (relu, etc.)

import torch.nn as nn
import torch.nn.functional as F

#Inherits from PyTorch’s nn.Module
#Your actual neural network
class OCRNet(nn.Module):
    def __init__(self):
        super(OCRNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # [B, 32, 32, 32]
        self.pool1 = nn.MaxPool2d(2, 2)                          # [B, 32, 16, 16]
        '''
        conv1:
        Input channels = 1 (grayscale)
        Output channels = 32 (feature maps)
        Padding = 1 (to preserve size)

        pool1:
        Max pooling 2×2 → halves width and height
        32×32 becomes 16×16
        '''

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # [B, 64, 16, 16]
        self.pool2 = nn.MaxPool2d(2, 2)                           # [B, 64, 8, 8]
        '''
        conv2:
        Input channels = 32 (from conv1)
        Output channels = 64 (more feature maps)

        pool2:
        Max pooling 2×2 → halves width and height
        16×16 becomes 8×8
        '''

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 62)  # 62 output classes
        '''
        fc1: Dense layer from 64×8×8 (4096) → 128
        fc2: Output layer → 62 units (one per character class)
        '''

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    '''
    Step-by-step:

    Conv1 → ReLU → Pool → [B, 32, 16, 16]
    Conv2 → ReLU → Pool → [B, 64, 8, 8]
    Flatten to [B, 4096]
    FC → ReLU → FC → Output logits
    '''



'''
### 📌 Where Are the 1024 Input Features?

Our 32×32 grayscale image contains **1024 pixel intensity values**, 
but since we are using a **Convolutional Neural Network (CNN)**, 
these values are **not flattened directly into a 1D vector at the beginning** 
(as in traditional fully connected networks). 
Instead, they are passed as a structured 2D input (`[1, 32, 32]`) to the convolutional layers, 
which preserve spatial relationships.

Each convolutional layer extracts meaningful patterns 
(like strokes, curves, edges) using small filters. 
Only after multiple convolution and pooling steps do we flatten 
the output to feed into the fully connected layers. 
So, the **1024 input values are still used**, 
but in a spatially-aware manner — allowing the CNN to learn better from the 
structure of the character image.

'''