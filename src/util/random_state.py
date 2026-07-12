import torch
import numpy as np
import random


def set_local_random_state_current_pipeline(
    random_seed=42, is_torch_pipeline: bool = True
):
    random.seed(random_seed)
    if is_torch_pipeline:
        torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def fix_torch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_torch_seed()
