import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod
from Config.config import TensorType, ShapeType


class Model_Base(torch.nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        name: str = "Model_Base",
        input_shape: ShapeType = (1, 3, 64, 64),
        num_classes: int = 2,
    ):
        super().__init__()
        self.name = name
        self.input_shape = input_shape
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: TensorType = None) -> TensorType:
        assert x is None, "Input tensor X can't be None!"
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.forward.__name__)
        )
