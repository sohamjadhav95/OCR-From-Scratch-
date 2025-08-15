import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# This class inherits from PyTorch's Dataset, which allows it to work with DataLoader.
class OCRDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        # Define transforms with augmentations for better generalization
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), shear=5),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ], p=0.7),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # normalize to [-1, 1]
        ])

        # Scan all folders
        folders = sorted(os.listdir(root_dir))
        for label_index, folder in enumerate(folders):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder_path, file))
                    self.labels.append(label_index)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
