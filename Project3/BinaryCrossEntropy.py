# imports
import numpy as np
from Layer import Layer


class BinaryCrossEntropy(Layer):
    def __init__(self):
        self.targets = None
        self.prediction = None

    def forward(self, prediction, targets):
        self.prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        self.targets = targets

        loss = -(self.targets * np.log(self.prediction) + (1 - self.targets) * np.log(1 - self.prediction))
        return np.mean(loss)

    def backward(self, upstream_gradients=1):
        batch_size = self.targets.shape[0]
        gradients = (self.prediction - self.targets) / (self.prediction * (1 - self.prediction))
        return (upstream_gradients / batch_size) * gradients
