"""CIFAR-10 data and representation utilities for BNN-DFA-QUBO."""
from utils.CIFAR10.data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_cifar10_datasets,
    get_cifar10_loaders,
    get_device,
)

__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "get_cifar10_datasets",
    "get_cifar10_loaders",
    "get_device",
]
