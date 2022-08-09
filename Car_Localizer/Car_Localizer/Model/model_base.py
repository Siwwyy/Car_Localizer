import torch
import torch.nn as nn

from abc import ABCMeta, abstractmethod

from Config.config import TensorType, ShapeType


class Model_Base(torch.nn.Module, metaclass=ABCMeta):
    """
    Base class with abstract model of our Models (derives from torch.nn.Module and metaclass=ABCMeta)

    Attributes
    ----------
    name : str
        name of model
    input_shape : ShapeType (look at possible shape types in config.py file)
        input shape of tensors
    num_classes : int
        number of classes in prediction
    **kwargs : Any
        keyword arguments (currently not used)

    Methods
    -------
    forward(self, x: TensorType = None) -> TensorType:
        Abstract Forward propagation's method of model
    """

    def __init__(
        self,
        name: str = "Model_Base",
        input_shape: ShapeType = (1, 3, 64, 64),
        num_classes: int = 2,
        **kwargs
    ):
        assert len(input_shape) > 2, "Input shape should contain at least CHW dims"
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
