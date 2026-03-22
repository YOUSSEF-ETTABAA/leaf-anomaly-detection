"""
autoencoder.py
--------------
The core deep learning model of this project.

Architecture: Vanilla MLP Autoencoder (Undercomplete Feedforward)

How it works:
    Input (100 dims)
        ↓
    Encoder: compress to smaller representation
        ↓
    Latent space (128 dims) ← bottleneck
        ↓
    Decoder: reconstruct back to original size
        ↓
    Output (100 dims)

Training strategy:
    - Train ONLY on healthy leaf features
    - Loss = difference between input and reconstruction (MSE)
    - After training: healthy leaves reconstruct well (low error)
                      diseased leaves reconstruct badly (high error)
    - Reconstruction error = anomaly score

Why MLP and not CNN?
    We work on DINOv2 feature vectors (100 numbers),
    not on raw images. So we don't need convolutional layers.
    The heavy visual processing was already done by DINOv2.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# ── Encoder ──────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Compresses the 100-dim input down to 128-dim latent space.

    Each layer does:
        Linear → BatchNorm → ReLU → Dropout
    BatchNorm stabilizes training.
    Dropout prevents overfitting (randomly turns off 20% of neurons).
    """
    def __init__(self, input_dim=100, latent_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1: expand then compress
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 2: compress to latent space
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


# ── Decoder ──────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Reconstructs the original 100-dim vector from the latent space.

    Mirror of the encoder — expands back to original size.
    No activation on the last layer (output can be any value).
    """
    def __init__(self, latent_dim=128, output_dim=100):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1: expand from latent space
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 2: reconstruct to original size
            nn.Linear(256, output_dim),
            # No activation here — output can be negative
        )

    def forward(self, x):
        return self.network(x)


# ── Full Autoencoder ──────────────────────────────────────────

class Autoencoder(nn.Module):
    """
    Full Autoencoder = Encoder + Decoder

    Training:
        - Feed healthy leaf features
        - Compute MSE between input and reconstruction
        - Minimize loss → model learns to reconstruct healthy leaves

    Inference (anomaly detection):
        - Feed any leaf feature
        - Compute reconstruction error
        - High error → anomaly (the model can't reconstruct it well)
        - Low error  → healthy (looks like what it was trained on)
    """
    def __init__(self, input_dim=100, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        """Standard forward pass: encode then decode."""
        compressed    = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed

    def anomaly_score(self, x):
        """
        Computes reconstruction error for each sample.
        Higher score = more anomalous.

        Args:
            x: torch.Tensor of shape (N, input_dim)

        Returns:
            numpy array of shape (N,) with anomaly scores
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Mean Squared Error per sample
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error.cpu().numpy()


# ── Model utilities ───────────────────────────────────────────

def save_model(model, path):
    """Saves model weights to disk."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  Model saved to: {path}")


def load_model(path, input_dim=100, latent_dim=128, device=None):
    """
    Loads a trained autoencoder from disk.

    Args:
        path       : path to .pth file
        input_dim  : must match training config
        latent_dim : must match training config
        device     : cuda, cpu, or None (auto-detect)

    Returns:
        model in eval mode, device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    print(f"  Autoencoder loaded from: {path}")
    return model, device


def compute_scores(model, features, device, batch_size=256):
    """
    Computes anomaly scores for a large set of features.
    Processes in batches to avoid memory issues.

    Args:
        model      : trained Autoencoder
        features   : numpy array of shape (N, input_dim)
        device     : cuda or cpu
        batch_size : how many samples per batch

    Returns:
        numpy array of shape (N,) with anomaly scores
    """
    model.eval()
    tensor  = torch.FloatTensor(features).to(device)
    dataset = TensorDataset(tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_scores = []
    with torch.no_grad():
        for (batch,) in loader:
            reconstructed = model(batch)
            scores = torch.mean((batch - reconstructed) ** 2, dim=1)
            all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores)
