class Layer:

    def __init__(self):
        self.weights = None
        self.bias = None
        self.gradients = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradients):
        raise NotImplementedError
