import random
import argparse

import numpy as np
import torch


def set_seed(seed=42):
    """
    Set the random seed for normal experiments.

    Controls:
        - Python random
        - NumPy
        - PyTorch CPU
        - PyTorch CUDA
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return seed


def get_seed():
    """
    Read the seed supplied through the command line.

    If no seed is supplied, default to 42.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    return args.seed