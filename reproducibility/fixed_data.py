import numpy as np
from torch.utils.data import Dataset


class FixedOrderDataset(Dataset):
    """
    Dataset wrapper that exposes a base dataset in a fixed index order.

    The wrapper is dataset-agnostic. It can be used with MNIST, Stroke,
    or any other PyTorch-compatible dataset.
    """

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)

        if self.indices.ndim != 1:
            raise ValueError("indices must be a 1D array")

        if len(self.indices) > 0:
            if self.indices.min() < 0:
                raise ValueError("indices cannot contain negative values")

            if self.indices.max() >= len(self.base_dataset):
                raise ValueError(
                    "indices contain values outside the base dataset"
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = int(self.indices[index])
        return self.base_dataset[real_index]