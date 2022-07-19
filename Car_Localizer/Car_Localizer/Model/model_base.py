import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod



class Model_Base(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def foo(self):
        print("DDD")