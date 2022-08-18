from re import T
from Base_loss_model.loss_model_base import Loss_Base, TensorType, ShapeType

import torch
import torch.nn as nn
import numpy as np


class Loss_First(Loss_Base):
    def __init__(
        self,
        name: str = "Loss_MSE",
    ):
        super().__init__(name)

    def forward(self, x: TensorType = None, y_pred: TensorType = None) -> TensorType:
        output = ((y_pred - x) ** 2).mean()
        return output
