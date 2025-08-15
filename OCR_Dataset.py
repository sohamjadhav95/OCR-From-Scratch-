from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import os
from torchvision import transforms
import random

class OCRDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.class_to_idx = {}

        # Allowed image extensions
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        # Create mapping and list of (image_path, label_idx)
        folders = sorted(os.listdir(root_dir))
        for idx, folder in enumerate(folders):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue  # Skip files in root_dir

            self.class_to_idx[folder] = idx
            for img_name in os.listdir(folder_path):
                ext = os.path.splitext(img_name)[1].lower()
                if ext in valid_exts:  # Only keep valid image files
                    self.samples.append((os.path.join(folder_path, img_name), idx))

        # Normal transform (no augmentation)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Augmentation transform
        self.augment_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),  # ±10°, ±5% shift
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('L')
        except (PermissionError, UnidentifiedImageError, IsADirectoryError, OSError) as e:
            print(f"⚠️ Skipping unreadable file: {img_path} ({e})")
            # Skip this index by picking a different one
            return self.__getitem__((idx + 1) % len(self.samples))

        # 50% chance to augment
        if random.random() > 0.5:
            image = self.augment_transform(image)
        else:
            image = self.transform(image)

        return image, label
