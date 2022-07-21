import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod
from typing import Union


TensorType = torch.tensor
ShapeType = Union[tuple, torch.Size]


class Model_Base(torch.nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        name: str = "Model_Base",
        input_shape: ShapeType = (1, 3, 64, 64),
        num_classes: int = 2,
    ):
        self.name = name
        self.input_shape = input_shape
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: TensorType = None) -> TensorType:
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.forward.__name__)
        )
