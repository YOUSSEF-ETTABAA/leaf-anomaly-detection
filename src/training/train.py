"""
train.py
--------
Training loop for the Autoencoder.

Key features:
    1. Trains only on healthy leaf features
    2. Saves checkpoints every N epochs (protection against crashes)
    3. Automatically resumes from checkpoint if one exists
    4. Tracks and returns loss history for visualization

Loss function: Mean Squared Error (MSE)
    - Measures how different the reconstruction is from the input
    - Lower loss = better reconstruction = model is learning

Optimizer: Adam
    - Adaptive learning rate optimizer
    - Works well for most neural networks without much tuning
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models.autoencoder import Autoencoder, save_model


def train_autoencoder(
    train_features,
    save_dir,
    input_dim        = 100,
    latent_dim       = 128,
    epochs           = 30,
    batch_size       = 64,
    learning_rate    = 0.001,
    checkpoint_every = 5,
    device           = None
):
    """
    Trains the autoencoder on healthy leaf features.

    Args:
        train_features   : numpy array of shape (N, input_dim)
        save_dir         : directory to save checkpoints and final model
        input_dim        : input/output size (= PCA components)
        latent_dim       : bottleneck size
        epochs           : number of training epochs
        batch_size       : samples per gradient update
        learning_rate    : how fast the model learns
        checkpoint_every : save checkpoint every N epochs
        device           : cuda or cpu (None = auto-detect)

    Returns:
        model         : trained Autoencoder
        loss_history  : list of average loss per epoch
    """
    # ── Setup ────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)
    print(f"Training on: {device}")
    print(f"Training samples: {len(train_features)}")

    # ── Create DataLoader ────────────────────────────────────
    # Convert numpy array to PyTorch tensor
    tensor  = torch.FloatTensor(train_features)
    dataset = TensorDataset(tensor)
    loader  = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle    = True   # shuffle for better training
    )

    # ── Build model ──────────────────────────────────────────
    model     = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()    # Mean Squared Error loss

    # ── Resume from checkpoint if exists ─────────────────────
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found — resuming from last save...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"  Resuming from epoch {start_epoch}")

    # ── Training loop ─────────────────────────────────────────
    loss_history = []

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(device)

            # Forward pass: compute reconstruction
            reconstructed = model(batch)

            # Compute loss: how different is reconstruction from input?
            loss = criterion(reconstructed, batch)

            # Backward pass: update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss for this epoch
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        print(f"  Epoch [{epoch+1:3d}/{epochs}]  Loss: {avg_loss:.6f}")

        # Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_every == 0:
            torch.save({
                "epoch"                : epoch,
                "model_state_dict"     : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "loss"                 : avg_loss,
            }, checkpoint_path)
            print(f"    Checkpoint saved at epoch {epoch+1}")

    # ── Save final model ──────────────────────────────────────
    final_path = os.path.join(save_dir, "autoencoder_final.pth")
    save_model(model, final_path)
    print(f"\nTraining complete!")
    print(f"  Final loss : {loss_history[-1]:.6f}")
    print(f"  Model saved: {final_path}")

    return model, loss_history


def plot_training_loss(loss_history, save_path=None):
    """
    Plots the training loss curve.
    Useful to check if the model converged properly.

    Args:
        loss_history : list of loss values per epoch
        save_path    : if provided, saves the plot
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history,
             color='#4CAF50', linewidth=2, marker='o', markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss", fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Loss curve saved to: {save_path}")

    plt.show()
