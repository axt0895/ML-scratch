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
            self.input = layer.forward(input)
        return input

    def backward(self, output_gradients, learning_rate = 0.001):
        for layer in self.layers:
            output_gradients = layer.backward(output_gradients, learning_rate)
        return output_gradients


# Create a Sequential model
model = Sequential()

# Add layers
model.add(Linear())
model.add(Sigmoid())


input_data = np.random.randn(1, 100)
output = model.forward(input_data)

output_gradient = np.random.randn(1, 2)
model.backward(output_gradient, learning_rate=0.001)