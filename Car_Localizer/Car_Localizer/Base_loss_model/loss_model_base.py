import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod
from Config.config import TensorType


class Loss_Base(torch.nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        name: str = "Loss_MSE",
    ):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, x: TensorType = None, y_pred: TensorType = None) -> TensorType:
        assert x is None, "Input tensor X can't be None!"
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.forward.__name__)
        )
