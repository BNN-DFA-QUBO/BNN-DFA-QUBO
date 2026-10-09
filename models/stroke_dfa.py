import torch


class StrokeDFA:
    def __init__(self, hidden_size, device):
        self.feedback = torch.randn(
            1,
            hidden_size,
            device=device
        )

    def hidden_gradient(self, output_error):
        return output_error @ self.feedback


class StrokeDFAFunction:
    @staticmethod
    def output_error(logits, labels):
        probabilities = torch.sigmoid(logits)
        # Gradient of mean BCEWithLogitsLoss; the caller divides by batch size
        # before forming weight and bias gradients.
        return probabilities - labels
