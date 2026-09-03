import torch
import torch.nn as nn


class DFAClassifier:
    """
    Direct Feedback Alignment for a two-layer network.

    The output error is projected directly to the hidden layer
    through a fixed random feedback matrix.
    """

    def __init__(self, hidden_size, output_size, device):
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Fixed random feedback matrix.
        # This matrix is NOT trained.
        self.feedback = torch.randn(
            output_size,
            hidden_size,
            device=device
        )

    def hidden_gradient(self, output_error):
        """
        Project output error directly to the hidden layer.

        output_error:
            [batch_size, output_size]

        returns:
            [batch_size, hidden_size]
        """

        return output_error @ self.feedback


class DFAFunction:
    """
    Utility functions for manually computing DFA updates.
    """

    @staticmethod
    def output_error(outputs, labels):
        """
        Compute gradient of cross-entropy loss with respect
        to the output logits.
        """

        probabilities = torch.softmax(outputs, dim=1)

        targets = torch.zeros_like(probabilities)

        targets.scatter_(
            1,
            labels.unsqueeze(1),
            1.0
        )

        return (probabilities - targets) / labels.size(0)