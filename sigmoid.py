import numpy as np
import random


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class MLP(object):
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input + 1 #bias
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.a_input = [1.0] * self.n_input
        self.a_hidden = [1.0] * self.n_hidden
        self.a_output = [1.0] * self.n_output

        self.w_input = np.zeros((len(self.n_input), len(self.n_hidden)))
        self.w_output = np.zeros((len(self.n_hidden), len(self.n_output)))

        for i in range(self.n_input):
            for j in range(self.n_hidden):
                self.w_input[i][j] = random.uniform(-1, 1)

        for k in range(self.n_hidden):
            for l in range(self.n_output):
                self.w_output[k][l] = random.uniform(-1, 1)

        self.c_input = np.zeros((len(self.n_input), len(self.n_hidden)))
        self.c_output = np.zeros((len(self.n_hidden), len(self.n_output)))

    def update(self, inputs):
        if len(inputs) != self.n_input - 1:
            raise ValueError('wrong inputs')

        for i in range(self.n_input - 1):
            self.a_input[i] = inputs[i]

        for j in range(self.n_hidden):
            _sum = 0.0
            for k in range(self.n_input):
                _sum += self.a_input[i] * self.w_input[k][j]
            self.a_hidden[j] = sigmoid(sum)

        for l in range(self.n_output):
            _sum = 0.0
            for n in range(self.n_hidden):
                _sum += self.a_hidden[n] * self.w_output[n][l]
            self.a_output[l] = sigmoid(_sum)

        return self.a_output[:]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.n_output:
            raise ValueError('wrong target values')

        output_deltas = [0.0] * self.n_output
        for k in range(self.n_output):
            error = targets[k] - self.a_output[k]
        