import numpy as np
from Layer import Layer


class ReLU(Layer):

    def __init__(self):
        super().__init__()
        self.input = input

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, gradients):
        return gradients * (self.input > 0)
