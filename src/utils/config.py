"""
config.py
---------
Loads settings from config.yaml and makes them available
to all other modules.

Usage:
    from src.utils.config import load_config
    cfg = load_config()
    print(cfg['autoencoder']['epochs'])  # 30
"""

import yaml
import os


def load_config(config_path=None):
    """
    Loads config.yaml from the project root.

    Args:
        config_path: optional path to config file.
                     If None, looks for config.yaml in current directory.

    Returns:
        dict with all configuration values
    """
    if config_path is None:
        # Look for config.yaml starting from current directory
        # then going up to find the project root
        search_path = os.getcwd()
        for _ in range(5):  # search up to 5 levels up
            candidate = os.path.join(search_path, 'config.yaml')
            if os.path.exists(candidate):
                config_path = candidate
                break
            search_path = os.path.dirname(search_path)

    if config_path is None or not os.path.exists(config_path):
        raise FileNotFoundError(
            "config.yaml not found. Make sure you're running from the project root."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_paths(config, base_path):
    """
    Resolves all relative paths in config to absolute paths.

    Args:
        config    : loaded config dict
        base_path : project root path (e.g. /content/drive/MyDrive/leaf-anomaly-detection)

    Returns:
        dict with all absolute paths
    """
    paths = config['paths']

    return {
        'train_healthy'  : os.path.join(base_path, paths['data']['train_healthy']),
        'test_healthy'   : os.path.join(base_path, paths['data']['test_healthy']),
        'test_anomaly'   : os.path.join(base_path, paths['data']['test_anomaly']),
        'cache_features' : os.path.join(base_path, paths['cache']['features']),
        'cache_pca'      : os.path.join(base_path, paths['cache']['pca']),
        'models'         : os.path.join(base_path, paths['models']),
        'outputs'        : os.path.join(base_path, paths['outputs']),
    }
