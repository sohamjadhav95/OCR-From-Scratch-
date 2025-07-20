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
        '''root_dir: Folder that contains subfolders (e.g., Sample001 to Sample062)
        image_paths: List of full file paths to each image
        labels: Corresponding label (e.g., 0 for Sample001, 1 for Sample002, etc.)'''

        # Define transforms: Convert to tensor and normalize
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),  # converts to [0,1] float tensor
        ])
        '''Ensure image is grayscale (1 channel)
        Resize to 32×32 pixels
        Convert to a tensor of shape [1, 32, 32] with pixel values between 0–1'''

        # Scan all folders
        folders = sorted(os.listdir(root_dir))
        for label_index, folder in enumerate(folders):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            '''Gets all subfolders inside root_dir and sorts them alphabetically
            enumerate() gives:
            label_index = 0 → for Sample001
            label_index = 61 → for Sample062
            This maps folder names to numeric labels.'''

            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder_path, file))
                    self.labels.append(label_index)  # label from folder index
            '''Scans each folder
            Filters image files
            Appends full file path to self.image_paths
            Assigns a label from label_index'''

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = self.labels[idx]
        '''Loads image from disk using index
        Applies transforms (resize, grayscale, tensor)
        Returns: a tuple of (image_tensor, label_int)'''

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


"""
Summary:

| Method        | Purpose                                     |
| ------------- | ------------------------------------------- |
| `__init__`    | Collects all image paths and labels         |
| `__len__`     | Tells DataLoader how many samples           |
| `__getitem__` | Returns one transformed image and its label |

"""