import numpy as np

class Layer:
    def forward(self, x): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def update(self, lr): pass 

class FullyConnected(Layer):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)
    
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0)
        return grad @ self.W.T

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask

class BatchNorm(Layer):
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def forward(self, x):
        self.x = x
        if x.ndim == 2:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
        else:
            raise NotImplementedError("Only supports 2D inputs")
        self.mean, self.var = mean, var
        self.x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad):
        N = self.x.shape[0]
        x_mu = self.x - self.mean
        std_inv = 1. / np.sqrt(self.var + self.eps)

        dxhat = grad * self.gamma
        dvar = np.sum(dxhat * x_mu, axis=0) * -0.5 * std_inv**3
        dmean = np.sum(dxhat * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)

        dx = (dxhat * std_inv) + (dvar * 2 * x_mu / N) + (dmean / N)
        self.dgamma = np.sum(grad * self.x_hat, axis=0)
        self.dbeta = np.sum(grad, axis=0)
        return dx

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta

class SoftmaxWithLoss(Layer):
    def forward(self, x, y):
        self.y = y
        x = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        loss = -np.log(self.probs[np.arange(len(y)), y] + 1e-9).mean()
        return loss

    def backward(self, _=None):
        dx = self.probs
        dx[np.arange(len(self.y)), self.y] -= 1
        return dx / len(self.y)
