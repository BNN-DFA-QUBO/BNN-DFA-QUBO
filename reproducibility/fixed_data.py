import numpy as np
from torch.utils.data import Dataset


class FixedOrderDataset(Dataset):

    def __init__(self, base_dataset, indices):

        self.base_dataset = base_dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        real_index = int(self.indices[index])

        return self.base_dataset[real_index]