from re import T
from Base_loss_model.loss_model_base import Loss_Base, TensorType, ShapeType

import torch
import torch.nn as nn
import numpy as np


class Loss_First(Loss_Base):
    def __init__(
        self,
        name: str = "Loss_First",
        input_shape: ShapeType = (1, 3, 64, 64),
        num_classes: int = 2,
    ):
        super().__init__(name, input_shape, num_classes)

    def forward(self, x: TensorType = None, y_pred: TensorType = None) -> TensorType:
        output = np.square(np.substract(x, y_pred)).mean()
        return output
