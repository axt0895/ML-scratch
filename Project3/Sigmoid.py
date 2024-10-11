import numpy as np

from Layer import Layer


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-self.inputs) + 1e-10)
        return self.output

    def backward(self, gradients):
        sigmoid = self.output
        return gradients * sigmoid * (1 - sigmoid)
