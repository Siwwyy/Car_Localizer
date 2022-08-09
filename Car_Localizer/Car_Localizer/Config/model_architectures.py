from typing import Tuple, List, Union


yolo_v1_architecture_config: List[Union[Tuple, str, List]] = [
    (7, 64, 2, 3),
    "M",  # MaxPool layer
    (3, 192, 1, 1),
    "M",  # MaxPool layer
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",  # MaxPool layer
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",  # MaxPool layer
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]
""" 
    Config and architecture structure of Yolo v1 model
    @TODO -> Add a config model preparation in futre based on Yaml file with OmegaConf lib
"""
