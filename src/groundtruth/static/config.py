"""
Configuration file.
"""

import os

VOCAB_SIZE = 5000
USE_CUDA = os.environ.get("PALMTREE_USE_CUDA", "1").lower() not in {"0", "false", "no"}
DEVICES = [
    int(device.strip())
    for device in os.environ.get("PALMTREE_CUDA_DEVICES", "0").split(",")
    if device.strip()
]
CUDA_DEVICE = int(os.environ.get("PALMTREE_CUDA_DEVICE", DEVICES[0] if DEVICES else 0))
VERSION = 1
MAXLEN = 10

LEARNING_RATE=1e-5
