"""Common utility helpers for logging, seeding, and feature loading."""

import logging
import os
import torch
import numpy as np
import random


def load_embedding(filename):
    """Load a single embedding file from disk."""
    if not os.path.exists(filename):
        print(f"Error: File not found at {filename}")
        return None
    return torch.load(filename)

def setup_seed(seed):  # Set random seeds.
    """Set random seeds for reproducible experiments."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(txt_path, mode='a+'):
    """Create a file-and-console logger for experiment output."""
     # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode=mode)  # w
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
