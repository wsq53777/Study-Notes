import numpy as np
from utils import load_mnist
from network import SimpleNet

def accuracy(pred, label):
    return np.mean(pred == label)

def train():
    (x_train, y_train), (x_val, y_val), _ = load_mnist()
    net = SimpleNet()
    batch_size = 64
    epochs = 10
    lr = 0.01

    for epoch in range(epochs):
        idx = np.random.permutation(len(x_train))
        x_train, y_train = x_train[idx], y_train[idx]

        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            loss = net.forward(x_batch, y_batch)
            net.backward()
            net.update(lr)

        val_pred = net.predict(x_val)
        acc = accuracy(val_pred, y_val)
        print(f"Epoch {epoch+1}, Val Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
