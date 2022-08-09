from __future__ import annotations

import torch

from typing import Union
from typing import Annotated

# TensorType = Annotated[torch.tensor, "Possible Tensor type"]
# ShapeType = Annotated[Union[tuple, torch.Size], "Possible Shape types of the tensor"]


TensorType = torch.tensor
""" 
    Possible Tensor type
"""

ShapeType = Union[tuple, torch.Size]
""" 
    Possible Shape types of the tensor
"""


def try_gpu(gpu_idx: int = 0) -> torch.device:
    """Returns a device (GPU) on specified gpu_idx,
        If GPU does not exist, then it returns CPU

    Parameters
    ----------
    gpu_idx : int
        GPU Device idx, number of specified GPU

    Returns
    -------
        function returns GPU on specified index if exists, if not, CPU."""
    if torch.cuda.device_count() >= gpu_idx + 1:
        return torch.device(f"cuda:{gpu_idx}")
    return torch.device("cpu")


UsedDevice: torch.device = try_gpu(gpu_idx=0)
""" 
    Currently used device
"""
