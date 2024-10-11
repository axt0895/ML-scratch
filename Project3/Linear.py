import numpy as np
from Layer import Layer

class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.weights + self.bias

    def backward(self, gradients):
        self.gradients = self.inputs @ gradients
        self.bias = np.sum(gradients, axis = 0)
        return np.dot(gradients, self.weights.T)
