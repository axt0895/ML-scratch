class Layer:

    def __init__(self):
        self.weights = None
        self.bias = None
        self.gradients = None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, gradients):
        raise NotImplementedError
