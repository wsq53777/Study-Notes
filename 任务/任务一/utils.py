import gzip
import pickle
import numpy as np

def load_mnist(path='mnist.pkl.gz'):
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, val_set, test_set

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]
