import torch

from utils.seed import set_seed


DEFAULT_SEED = 42


def set_deterministic(seed=DEFAULT_SEED):
    """
    Configure deterministic/reproducible execution.

    This builds on the normal project-wide seed function
    and additionally enables deterministic CUDA/cuDNN behavior.
    """

    set_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[REPRO] Seed = {seed}")