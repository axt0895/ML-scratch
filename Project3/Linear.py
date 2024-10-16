# Imports
import numpy as np
from Layer import Layer


# Class that implements linear layer. It implements forward and backward function.
class Linear(Layer):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.inputs = None

    def forward(self, inputs) -> np.ndarray:
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate=0.01) -> np.ndarray:
        weights_gradient = np.matmul(self.inputs.T, output_gradient)
        input_gradient = np.matmul(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        return input_gradient
