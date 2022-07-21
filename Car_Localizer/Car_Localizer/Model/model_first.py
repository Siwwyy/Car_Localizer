from Model.model_base import Model_Base, TensorType, ShapeType

import torch
import torch.nn as nn


class Model_First(Model_Base):
    def __init__(
        self,
        name: str = "Model_First",
        input_shape: ShapeType = (1, 3, 64, 64),
        num_classes: int = 2,
    ):
        super().__init__(name, input_shape, num_classes)

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # conv3 1x1 convolution as a Fully Connected layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            nn.Relu(),
        )

        # conv4 1x1 convolution as a Fully Connected layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            nn.Relu(),
        )

    def forward(self, x: TensorType = None) -> TensorType:

        # Input convolution
        output = self.conv1(x)

        # Second convolution
        output = self.conv2(output)

        # Third convolution, 1x1 -> Flatten Conv
        output = self.conv3(output)

        # Fourth convolution, 1x1 -> Flatten Conv
        output = self.conv4(output)
        return output
