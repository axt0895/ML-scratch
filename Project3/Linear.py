import numpy as np
from Layer import Layer
class Linear:
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))

    def forward(self, X):
        self.input = X
        return X @ self.weights + self.bias

    def backward(self, output, learning_rate = 0.01):
        pass