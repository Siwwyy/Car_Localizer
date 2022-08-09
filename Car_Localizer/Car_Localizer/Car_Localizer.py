import torch
import Model

from Model import Model_Yolo_v1

# a = Model.Model_First()


# Yolo_v1 test function
def test(split_size: int = 7, num_boxes: int = 2, num_classes: int = 20):
    model = Model_Yolo_v1(
        split_size=split_size, num_boxes=num_boxes, num_classes=num_classes
    )
    x = torch.randn((10, 3, 448, 448))

    # print(model)

    print(model(x).shape)


test()
