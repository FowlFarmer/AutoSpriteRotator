from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import csv
from PIL import Image

class TransformLabelDataset(Dataset):
    def __init__(self, images_dir, label_path):
        """
        image_paths: list of file paths or preloaded tensors
        labels: list of dicts with 'flip', 'rot', 'scale'
        transform: optional torchvision transform or custom transform
        """
        self.transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),  # Converts PIL → [C,H,W] float tensor in [0.0, 1.0]
        # Normalization done in the model, not here
        ])
        self.images = []
        self.labels = []
        with open(label_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            header = next(reader, None) # Skip header row
            for row in reader:
                filename = row['filename']
                image_path = os.path.join(images_dir, filename)
                if os.path.exists(image_path):
                    self.images.append(image_path)
                    self.labels.append({
                        "flip": int(row['flip']), # torch casts to float correctly
                        "rot": float(row['rotation_radians']),
                        "scale": float(row['scale'])
                    })
                else:
                    print(f"Warning: Image {image_path} does not exist, skipping.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # 🖼️ Load the image as RGBA (for 4 channels)
        img = Image.open(img_path).convert("RGBA")

        # 📐 Apply transforms (e.g., Resize, ToTensor)
        img = self.transform(img)

        return (
            img,  # shape [4, H, W]
            # torch.tensor([label["flip"]], dtype=torch.float32),
            torch.tensor([label["rot"]], dtype=torch.float32),
            torch.tensor([label["scale"]], dtype=torch.float32),
        )