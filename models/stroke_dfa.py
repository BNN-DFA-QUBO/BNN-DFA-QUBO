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
    def output_error(logits, labels, pos_weight):
        probabilities = torch.sigmoid(logits)

        error = probabilities - labels

        error = torch.where(
            labels == 1,
            error * pos_weight,
            error
        )

        return error