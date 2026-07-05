import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    return device


def create_output_dirs(*directories):
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
