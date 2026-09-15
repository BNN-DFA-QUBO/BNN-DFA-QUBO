import random
import numpy as np
import torch


SEED = 42


def set_deterministic(seed=SEED):

    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)

    # CUDA random
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Make CUDA/cuDNN operations as deterministic as possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[REPRO] Seed = {seed}")