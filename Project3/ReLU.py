import numpy as np

from Layer import Layer

class ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, gradients):
        return gradients * (self.inputs > 0)