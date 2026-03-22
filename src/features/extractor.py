"""
extractor.py
------------
Uses DINOv2 to extract feature vectors from leaf images.

What is DINOv2?
    A powerful Vision Transformer (ViT) pretrained by Meta.
    It converts any image into a 384-dimensional feature vector
    that captures the visual content of the image.

Why use it instead of training from scratch?
    DINOv2 was trained on 142 million images.
    It already understands shapes, textures, and patterns.
    We just use it as a "smart encoder" for our leaf images.

What does feature extraction produce?
    Input  : folder of images (e.g. 2546 healthy leaves)
    Output : numpy array of shape (2546, 384)
             each row = one image's feature vector

These features are then used by:
    - PCA      (for dimensionality reduction)
    - KNN      (for anomaly scoring)
    - Autoencoder (for anomaly scoring)
"""

import torch
import timm
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data.loader import LeafDataset


def load_dino_model(model_name="vit_small_patch14_reg4_dinov2.lvd142m"):
    """
    Loads the pretrained DINOv2 Vision Transformer.

    num_classes=0 removes the classification head —
    we only want the feature vectors, not class predictions.

    Args:
        model_name: timm model identifier

    Returns:
        model in eval mode, device (cuda or cpu)
    """
    print("Loading DINOv2 model...")

    model = timm.create_model(
        model_name,
        pretrained  = True,
        num_classes = 0    # remove classification head
    )
    model.eval()

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    print(f"  DINOv2 loaded on: {device}")
    return model, device


def extract_features(folder_path, model, device, batch_size=32, image_size=518):
    """
    Extracts DINOv2 feature vectors for all images in a folder.

    Process:
        1. Load images in batches
        2. Pass each batch through DINOv2
        3. Collect output vectors
        4. Stack into one numpy array

    Args:
        folder_path : path to image folder
        model       : loaded DINOv2 model
        device      : cuda or cpu
        batch_size  : images per batch (reduce if out of memory)
        image_size  : input size (518 for DINOv2)

    Returns:
        numpy array of shape (N, 384) where N = number of images
    """
    dataset = LeafDataset(folder_path, image_size=image_size)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True
    )

    all_features = []

    # torch.no_grad() saves memory — we don't need gradients here
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Extracting features"):
            batch    = batch.to(device)
            features = model(batch)                    # shape: (batch, 384)
            all_features.append(features.cpu().numpy())

    # Stack all batches into one array
    return np.concatenate(all_features, axis=0)        # shape: (N, 384)
