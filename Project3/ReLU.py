# Imports
import numpy as np
from Layer import Layer

# Implementation of ReLU Activation Layer
class ReLU(Layer):

    def __init__(self) -> None:
        super().__init__()
        self.input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.maximum(0, input)

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        return gradients * (self.input > 0)
