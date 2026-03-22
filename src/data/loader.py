"""
loader.py
---------
Handles loading images from folders and converting them
into PyTorch tensors ready for the model.

Two main things here:
1. LeafDataset   → a PyTorch Dataset that loads images from a folder
2. get_dataloader → creates a DataLoader (batches + parallel loading)

Why do we need transforms?
    DINOv2 expects images in a specific format:
    - Size: 518x518 pixels
    - Normalized with ImageNet mean and std
    These transforms handle that automatically.
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── Image transforms ─────────────────────────────────────────

def get_transforms(image_size=518):
    """
    Standard transforms applied to every image.

    Steps:
        1. Resize to 518x518 (required by DINOv2)
        2. Convert pixel values to 0-1 range (ToTensor)
        3. Normalize with ImageNet statistics
           (because DINOv2 was pretrained on ImageNet)

    Args:
        image_size: target size (default 518 for DINOv2)

    Returns:
        torchvision transforms pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean
            std =[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])


# ── Dataset class ─────────────────────────────────────────────

class LeafDataset(Dataset):
    """
    Loads all images from a folder.

    Works for any of our three data splits:
        - train/healthy
        - test/healthy
        - test/anomaly

    Args:
        folder_path : path to the image folder
        image_size  : resize target (default 518)
    """

    # Accepted image file extensions
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    def __init__(self, folder_path, image_size=518):
        self.folder_path = folder_path
        self.transform   = get_transforms(image_size)

        # Collect all valid image paths
        self.image_paths = sorted([
            p for p in Path(folder_path).iterdir()
            if p.suffix in self.VALID_EXTENSIONS
        ])

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in: {folder_path}")

        print(f"  Found {len(self.image_paths)} images in {folder_path}")

    def __len__(self):
        """Returns total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads one image by index and applies transforms.

        Returns:
            tensor of shape (3, 518, 518)
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


# ── DataLoader factory ────────────────────────────────────────

def get_dataloader(folder_path, image_size=518, batch_size=32, num_workers=2):
    """
    Creates a DataLoader for a folder of images.

    Args:
        folder_path : path to image folder
        image_size  : resize target
        batch_size  : images per batch
        num_workers : parallel loading workers
                      (use 0 on Windows to avoid multiprocessing issues)

    Returns:
        PyTorch DataLoader
    """
    dataset = LeafDataset(folder_path, image_size)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,   # no shuffle for feature extraction
        num_workers = num_workers,
        pin_memory  = True     # faster GPU transfer
    )
