"""
pca.py
------
Reduces DINOv2 features from 384 dimensions to 100 dimensions.

Why do we need PCA?
    DINOv2 outputs 384-dimensional vectors.
    Many of those dimensions contain noise or redundant information.
    PCA keeps only the most important 100 dimensions.

    This improves:
        - Distance calculations (KNN, Mahalanobis)
        - Autoencoder training speed
        - Overall anomaly detection performance

Important rule:
    PCA must be FITTED on training data only.
    Then APPLIED (transform) on both train and test data.
    Never refit on test data — that would be data leakage.

Usage:
    from src.features.pca import fit_pca, apply_pca, save_pca, load_pca
"""

import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_pca(train_features, n_components=100):
    """
    Fits StandardScaler + PCA on training features.

    Two steps:
        1. StandardScaler: normalize each feature dimension to mean=0, std=1
           This is important because PCA is sensitive to scale.
        2. PCA: find the n_components directions of maximum variance

    Args:
        train_features : numpy array of shape (N, 384)
        n_components   : how many dimensions to keep (default 100)

    Returns:
        reduced_features : numpy array of shape (N, 100)
        scaler           : fitted StandardScaler
        pca              : fitted PCA
    """
    print(f"  Fitting PCA: {train_features.shape[1]} → {n_components} dimensions")

    # Step 1: normalize
    scaler          = StandardScaler()
    features_scaled = scaler.fit_transform(train_features)

    # Step 2: reduce dimensions
    pca             = PCA(n_components=n_components, random_state=42)
    features_reduced = pca.fit_transform(features_scaled)

    # How much information is preserved
    variance_kept = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA keeps {variance_kept:.1f}% of information")
    print(f"  Shape after PCA: {features_reduced.shape}")

    return features_reduced, scaler, pca


def apply_pca(features, scaler, pca):
    """
    Applies already-fitted scaler and PCA to new data.

    Use this for test data — never refit on test data.

    Args:
        features : numpy array of shape (N, 384)
        scaler   : fitted StandardScaler from fit_pca()
        pca      : fitted PCA from fit_pca()

    Returns:
        numpy array of shape (N, n_components)
    """
    features_scaled  = scaler.transform(features)
    features_reduced = pca.transform(features_scaled)
    return features_reduced


def save_pca(scaler, pca, save_dir):
    """
    Saves scaler and PCA models to disk.

    Args:
        scaler   : fitted StandardScaler
        pca      : fitted PCA
        save_dir : directory to save files
    """
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(pca,    os.path.join(save_dir, "pca.pkl"))
    print(f"  PCA saved to: {save_dir}")


def load_pca(save_dir):
    """
    Loads scaler and PCA from disk.

    Args:
        save_dir : directory where files were saved

    Returns:
        scaler, pca
    """
    scaler = joblib.load(os.path.join(save_dir, "scaler.pkl"))
    pca    = joblib.load(os.path.join(save_dir, "pca.pkl"))
    print(f"  PCA loaded from: {save_dir}")
    return scaler, pca
