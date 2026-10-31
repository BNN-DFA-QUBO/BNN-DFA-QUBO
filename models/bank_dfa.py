"""Direct Feedback Alignment (DFA) model and error utilities for Bank dataset."""
import torch
import torch.nn as nn
from models.dfa import DFAClassifier, DFAFunction


class BankDFA(DFAClassifier):
    """DFA alignment engine for Bank dataset with 2 output classes and configurable hidden size."""

    def __init__(self, hidden_size: int = 64, output_size: int = 2, device: str | torch.device = "cpu"):
        super().__init__(hidden_size=hidden_size, output_size=output_size, device=device)


class BankDFAFunction(DFAFunction):
    """Gradient calculation utilities for Bank DFA softmax cross-entropy output error."""
    pass

