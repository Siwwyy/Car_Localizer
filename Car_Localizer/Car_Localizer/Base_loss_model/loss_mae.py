from Base_loss_model.loss_model_base import Loss_Base, TensorType
import torch


class Loss_MAE(Loss_Base):
    def __init__(
        self,
        name: str = "Loss_MAE",
    ):
        super().__init__(name)

    def forward(self, x: TensorType = None, y_pred: TensorType = None) -> TensorType:
        output = torch.abs(y_pred - x).mean()
        return output
