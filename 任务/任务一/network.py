from layers import *

class SimpleNet:
    def __init__(self):
        self.layers = [
            FullyConnected(784, 128),
            BatchNorm(128),
            ReLU(),
            FullyConnected(128, 10)
        ]
        self.loss_layer = SoftmaxWithLoss()

    def forward(self, x, y):
        for layer in self.layers:
            x = layer.forward(x)
        loss = self.loss_layer.forward(x, y)
        return loss

    def backward(self):
        grad = self.loss_layer.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return np.argmax(x, axis=1)
