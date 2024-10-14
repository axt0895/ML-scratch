import numpy as np

from Layer import Layer


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-self.input) + 1e-10)
        return self.output

    def backward(self, output_gradients, learning_rate = 0.001):
        sigmoid = self.output
        return output_gradients * sigmoid * (1 - sigmoid)
