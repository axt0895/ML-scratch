# Imports

import pickle
import numpy as np
from Layer import Layer
from Linear import Linear
from Sigmoid import Sigmoid
from ReLU import ReLU
from BinaryCrossEntropy import BinaryCrossEntropy

class Sequential(Layer):
    def __init__(self):
        super().__init__()
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_gradients, learning_rate = 0.01):
        for layer in reversed(self.layers):
            output_gradients = layer.backward(output_gradients, learning_rate)
        return output_gradients
