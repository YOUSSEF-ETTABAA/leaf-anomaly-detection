"""
cache.py
--------
Smart caching system to avoid recomputing expensive operations.

The idea is simple:
- First time  → compute the result, save it to disk
- Next time   → load from disk instead of recomputing

This is critical for DINOv2 feature extraction which takes
30+ minutes. With caching it becomes instant on the second run.

Usage:
    from src.utils.cache import load_or_compute
    import numpy as np

    features = load_or_compute(
        path       = 'cache/features/train.npy',
        compute_fn = lambda: extract_features(folder, model)
    )
"""

import os
import numpy as np


def load_or_compute(path, compute_fn, allow_pickle=False):
    """
    Loads a numpy array from cache if it exists,
    otherwise computes it and saves it.

    Args:
        path        : where to save/load the .npy file
        compute_fn  : function that computes the data (called only if cache miss)
        allow_pickle: whether to allow pickle in numpy load

    Returns:
        numpy array
    """
    if os.path.exists(path):
        print(f"  Loading from cache: {path}")
        return np.load(path, allow_pickle=allow_pickle)

    print(f"  Computing and saving: {path}")

    # Make sure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Compute the data
    data = compute_fn()

    # Save to disk
    np.save(path, data)
    print(f"  Saved to: {path}")

    return data


def cache_exists(path):
    """Check if a cache file already exists."""
    return os.path.exists(path)


def clear_cache(cache_dir):
    """
    Deletes all .npy files in a cache directory.
    Use this when you want to force recomputation.

    Args:
        cache_dir: path to cache directory
    """
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return

    deleted = 0
    for file in os.listdir(cache_dir):
        if file.endswith('.npy'):
            os.remove(os.path.join(cache_dir, file))
            deleted += 1

    print(f"Deleted {deleted} cache files from {cache_dir}")
