"""
prepare_data.py
---------------
Organizes the PlantVillage dataset into our project structure.

PlantVillage comes like this:
    PlantVillage/
        train/
            Tomato___healthy/         ← we use this for training
            Tomato___Bacterial_spot/  ← we use this for testing (anomaly)
            ...
        val/
            Tomato___healthy/         ← we use this for testing (healthy)
            Tomato___Bacterial_spot/  ← we use this for testing (anomaly)
            ...

We reorganize it into:
    data/
        train/
            healthy/   ← 80% of Tomato healthy train images
        test/
            healthy/   ← 20% of Tomato healthy + all val healthy
            anomaly/   ← all Tomato disease images (train + val)

Why this split?
    - Autoencoder trains ONLY on healthy images
    - Test set needs both healthy and anomaly to measure performance
"""

import os
import shutil
from pathlib import Path


def is_tomato(folder_name):
    """Check if a folder belongs to the Tomato plant."""
    return folder_name.startswith("Tomato")


def is_healthy(folder_name):
    """Check if a folder contains healthy leaf images."""
    return "healthy" in folder_name.lower()


def get_images(folder_path):
    """Get all image files from a folder."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = []
    for ext in extensions:
        images.extend(Path(folder_path).glob(ext))
    return sorted(images)


def prepare_dataset(source_path, dest_path, train_split=0.8, verbose=True):
    """
    Organizes PlantVillage into our train/test structure.

    Args:
        source_path : path to PlantVillage folder (contains train/ and val/)
        dest_path   : where to save organized data
        train_split : fraction of healthy images used for training (default 0.8)
        verbose     : print progress

    Returns:
        dict with counts of copied images
    """
    train_source = os.path.join(source_path, "train")
    val_source   = os.path.join(source_path, "val")

    # Create destination folders
    train_healthy_dest = os.path.join(dest_path, "train", "healthy")
    test_healthy_dest  = os.path.join(dest_path, "test",  "healthy")
    test_anomaly_dest  = os.path.join(dest_path, "test",  "anomaly")

    for folder in [train_healthy_dest, test_healthy_dest, test_anomaly_dest]:
        os.makedirs(folder, exist_ok=True)

    counts = {"train_healthy": 0, "test_healthy": 0, "test_anomaly": 0}

    # ── Process TRAIN split ──────────────────────────────────
    if verbose:
        print("Processing train split...")

    for folder_name in sorted(os.listdir(train_source)):
        if not is_tomato(folder_name):
            continue

        folder_path = os.path.join(train_source, folder_name)
        images = get_images(folder_path)

        if is_healthy(folder_name):
            # Split healthy: 80% train, 20% test
            split_idx    = int(len(images) * train_split)
            train_images = images[:split_idx]
            test_images  = images[split_idx:]

            for img in train_images:
                shutil.copy(str(img), train_healthy_dest)
                counts["train_healthy"] += 1

            for img in test_images:
                shutil.copy(str(img), test_healthy_dest)
                counts["test_healthy"] += 1

            if verbose:
                print(f"  {folder_name}: {len(train_images)} train, {len(test_images)} test")
        else:
            # All disease images go to anomaly test
            for img in images:
                shutil.copy(str(img), test_anomaly_dest)
                counts["test_anomaly"] += 1

            if verbose:
                print(f"  {folder_name}: {len(images)} anomaly")

    # ── Process VAL split ────────────────────────────────────
    if verbose:
        print("\nProcessing val split...")

    for folder_name in sorted(os.listdir(val_source)):
        if not is_tomato(folder_name):
            continue

        folder_path = os.path.join(val_source, folder_name)
        images = get_images(folder_path)

        if is_healthy(folder_name):
            # All val healthy go to test healthy
            for img in images:
                shutil.copy(str(img), test_healthy_dest)
                counts["test_healthy"] += 1

            if verbose:
                print(f"  {folder_name}: {len(images)} test healthy")
        else:
            # All val disease go to anomaly test
            for img in images:
                shutil.copy(str(img), test_anomaly_dest)
                counts["test_anomaly"] += 1

            if verbose:
                print(f"  {folder_name}: {len(images)} anomaly")

    # ── Summary ──────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*45}")
        print(f"  Dataset prepared successfully!")
        print(f"{'='*45}")
        print(f"  Train healthy : {counts['train_healthy']} images")
        print(f"  Test  healthy : {counts['test_healthy']} images")
        print(f"  Test  anomaly : {counts['test_anomaly']} images")
        print(f"{'='*45}")

    return counts
