import random
import MNIST
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.5
EPOCH = 10000
TRAINING_DATA = 0
TEST_DATA = 0
x1 = []
y1 = []

def sigmoid(x):
    return 1.0/(1.0+np.exp(-1.0 * x))


def sigmoid_prime(y):
    return y * (1.0 - y)


class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.ni = n_inputs + 1
        self.nh = n_hidden
        self.no = n_outputs
