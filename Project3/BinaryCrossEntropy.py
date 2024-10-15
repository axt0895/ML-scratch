# Imports
import numpy as np
from Layer import Layer


# Implementation of Binary Cross-Entropy Loss
class BinaryCrossEntropy(Layer):

    def __init__(self):
        super().__init__()
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        self.predictions = np.clip(predictions, 1e-7, 1- 1e-7)
        self.targets = targets

        loss = -(self.targets * np.log(self.predictions) + (1 - self.targets) * np.log(1 - self.predictions))
        return np.mean(loss)

    def backward(self, output_gradients = 1):
        batch_size = self.targets.shape[0]
        gradients = (self.predictions - self.targets) / (self.predictions * (1 - self.predictions))
        return output_gradients / batch_size * gradients