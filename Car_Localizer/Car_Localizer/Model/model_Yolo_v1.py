from Model.model_base import Model_Base, TensorType, ShapeType
from Config.model_architectures import yolo_v1_architecture_config as model_config
from typing import List

import torch
import torch.nn as nn


class CNNBlock(Model_Base):
    """
    CNNBlock, used to create a repeated CNN layers in Yolo v1 and other architectures (derives from Model_Base, look at model_base.py)

    Attributes
    ----------
    in_channels : int
        amount of input channels to conv block
    out_channels : int
        amount of outout channels of conv block
    **kwargs : Any
        keyword arguments for nn.Conv2d like: stride, padding etc.

    Methods
    -------
    forward(self, x: TensorType = None) -> TensorType:
        Forward propagation's method of CNN block
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, **kwargs):
        super().__init__()
        # Conv layer, bias = False for BatchNorm
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

        # BatchNorm layer
        self.batchnorm = nn.BatchNorm2d(out_channels)

        # Activation (LeakyReLU in that case)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x: TensorType = None) -> TensorType:
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Model_Yolo_v1(Model_Base):
    """
    Yolo_v1 model architecture implementation (derives from Model_Base, look at model_base.py)

    Attributes
    ----------
    name : str
        name of model
    input_shape : ShapeType (look at possible shape types in config.py file)
        input shape of tensors
    num_classes : int
        number of classes in prediction
    **kwargs : Any
        keyword arguments for fully connected layers at the end of Yolo_v1 architecture

    Methods
    -------
    forward(self, x: TensorType = None) -> TensorType:
        Forward propagation's method of model
    """

    def __init__(
        self,
        name: str = "Model_Yolo_v1",
        input_shape: ShapeType = (1, 3, 64, 64),
        num_classes: int = 2,
        **kwargs
    ):
        super().__init__(name, input_shape, num_classes)

        self.architecture = model_config
        self.in_channels = input_shape[-3]
        self.darknet = self._create_conv_layers(self.architecture)
        self.fully_connected = self._create_fully_connected(**kwargs)

    def forward(self, x: TensorType = None) -> TensorType:
        x = self.darknet(x)
        return self.fully_connected(
            torch.flatten(x, start_dim=1)
        )  # flattening through channels

    def _create_conv_layers(self, architecture: List = None) -> nn.Sequential:
        assert (
            architecture is not None
        ), "Architecture can't be None, pass architecture of specified model (e.g., yolo_v1_architecture_config in model_architectures.py config file)"

        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels,
                        out_channels=x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3],
                    )
                ]

                in_channels = x[
                    1
                ]  # in_channels is now output channels of last conv block

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                conv1 = x[0]  # Tuple with Conv architecture
                conv2 = x[1]  # Tuple with Conv architecture
                num_repeats = x[2]  # Int

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]

                    layers += [
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]

                    in_channels = conv2[
                        1
                    ]  # in_channels is now output channels of last conv block

        return nn.Sequential(*layers)  # unpack layers list here

    def _create_fully_connected(
        self, split_size: int = 7, num_boxes: int = 2, num_classes: int = 20
    ) -> nn.Sequential:
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                1024 * S * S, 496
            ),  # In original paper of Yolo_v1 should be 4096 instead of 496 output neurons
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),  # (S,S,30) -> shape: C + B * 5 = 30
        )


## Yolo_v1 test function
# def test(split_size:int=7, num_boxes:int=2, num_classes: int=20):
#    model = Model_Yolo_v1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
#    x = torch.randn((2,3,448,448))
#    print(model(x).shape)

# test()
