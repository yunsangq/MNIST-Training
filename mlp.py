import random
import numpy as np
import os
import struct
from array import array
import matplotlib.pyplot as plt


class MNIST(object):

    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels


def sigmoid(x):
    return 1.0/(1.0+np.exp(-1.0 * x))


def sigmoid_prime(y):
    return y * (1.0 - y)


class MLP(object):
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input + 1
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.act_input = [1.0] * self.n_input
        self.act_hidden = [1.0] * self.n_hidden
        self.act_output = [1.0] * self.n_output

        self.weight_input = np.zeros((len(self.act_input), len(self.act_hidden)))
        self.weight_output = np.zeros((len(self.act_hidden), len(self.act_output)))
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                self.weight_input[i][j] = random.uniform(-1.0, 1.0)

        for i in range(self.n_hidden):
            for j in range(self.n_output):
                self.weight_output[i][j] = random.uniform(-1.0, 1.0)

        self.x1 = []
        self.y1 = []

    def feed_forward(self, inputs):
        # act_input
        for i in range(self.n_input - 1):
            self.act_input[i] = inputs[i] / 256.0

        # act_hidden
        for j in range(self.n_hidden):
            _sum = 0.0
            for i in range(self.n_input):
                _sum += self.act_input[i] * self.weight_input[i][j]
            self.act_hidden[j] = sigmoid(_sum)

        # act_output
        for k in range(self.n_output):
            _sum = 0.0
            for j in range(self.n_hidden):
                _sum += self.act_hidden[j] * self.weight_output[j][k]
            self.act_output[k] = sigmoid(_sum)

        return self.act_output.index(max(self.act_output))

    def back_propagate(self, targets, n):
        # calculate error for output
        output_deltas = [0.0] * self.n_output
        for k in range(self.n_output):
            error = targets[k] - self.act_output[k]
            output_deltas[k] = sigmoid_prime(self.act_output[k]) * error

        # calculate error for hidden
        hidden_deltas = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            error = 0.0
            for k in range(self.n_output):
                error += output_deltas[k] * self.weight_output[j][k]
            hidden_deltas[j] = sigmoid_prime(self.act_hidden[j]) * error

        # update output weights
        for j in range(self.n_hidden):
            for k in range(self.n_output):
                change = output_deltas[k] * self.act_hidden[j]
                self.weight_output[j][k] += n * change

        # update input weights
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                change = hidden_deltas[j] * self.act_input[i]
                self.weight_input[i][j] += n * change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += (targets[k] - self.act_output[k]) ** 2
        error = 0.5 * np.sqrt(error) ** 2

        return error

    def training(self, patterns, outputs, time=5, learning_rate=0.5):
        for i in range(time):
            for j in range(len(outputs)):
                inputs = patterns[j]
                targets = np.zeros(10)
                pos = outputs[j]
                targets[pos] = 1.0
                self.feed_forward(inputs)
                error = self.back_propagate(targets, learning_rate)
                self.x1.append(j)
                self.y1.append(error)
                if j % 1000 == 0:
                    print ('time -> ' + str(j))
                    print ('error -> %-.5f' % error)
            print ('epoch -> ' + str(i))
            plt.plot(self.x1, self.y1)
            plt.xlabel('time')
            plt.ylabel('cost')
            plt.show()

    def test(self, patterns, labels):
        correct = 0.0
        for i in range(len(patterns)):
            result = self.feed_forward(patterns[i])
            if labels[i] == result:
                correct += 1.0
        print ('Correct = ' + str(correct))
        print ('Accuracy = ' + str(correct / 10000.0 * 100.0))

if __name__ == '__main__':
    mn = MNIST()
    img, label = mn.load_training()
    test_img, test_label = mn.load_testing()

    NN = MLP(784, 15, 10)
    NN.training(img, label)
    NN.test(test_img, test_label)
