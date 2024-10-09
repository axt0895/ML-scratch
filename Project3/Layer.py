class Layer:
    def forward(self, X, weights, bias):
        X = X @ weights + bias

    def backward(self):
        pass