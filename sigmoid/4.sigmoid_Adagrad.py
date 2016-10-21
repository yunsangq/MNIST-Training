import random
import numpy as np
import MNIST_Loader
import matplotlib.pyplot as plt
import time
import json


def err_disp(epochs, train, valid):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('sigmoid_Adagrad_Cost per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), train, color='#1F77B4', label='Training')
    ax.plot(range(epochs), valid, color='#b41f1f', label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def acc_disp(epochs, train, valid):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('sigmoid_Adagrad_Accuracy per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), train, color='#1F77B4', label='Training')
    ax.plot(range(epochs), valid, color='#b41f1f', label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


class MLP:
    def __init__(self):
        self.biases = [np.random.randn(y, 1) for y in NN[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(NN[:-1], NN[1:])]
        self.train_acc = []
        self.valid_acc = []
        self.train_cost = []
        self.valid_cost = []
        self.cache_b = [np.zeros(b.shape) for b in self.biases]
        self.cache_w = [np.zeros(w.shape) for w in self.weights]

    def feed_forward(self, inputs):
        for b, w in zip(self.biases, self.weights):
            inputs = self.sigmoid(np.dot(w, inputs) + b)
        return inputs

    def train(self, x, y):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        # Feed Forward
        activation = x
        activations = [x]

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = self.sigmoid(z)
            activations.append(activation)

        # Backpropagation
        delta = self._error(activations[-1], y) * self.dsigmoid(activations[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].transpose())

        for i in xrange(2, 3):
            delta = np.dot(self.weights[-i+1].transpose(), delta) * self.dsigmoid(activations[-i])
            delta_b[-i] = delta
            delta_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return delta_b, delta_w

    def update(self, mini_batch, learn_rate):
        batch_b = [np.zeros(b.shape) for b in self.biases]
        batch_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_batch_b, delta_batch_w = self.train(x, y)
            batch_b = [b + delta_b for b, delta_b in zip(batch_b, delta_batch_b)]
            batch_w = [w + delta_w for w, delta_w in zip(batch_w, delta_batch_w)]

        """
        cache += dx**2
        x += -lr*dx / (sqrt(cache) + eps)
        """

        self.cache_w = [cw + bw**2 for cw, bw in zip(self.cache_w, batch_w)]
        self.cache_b = [cb + bb**2 for cb, bb in zip(self.cache_b, batch_b)]

        self.weights = [w - (learn_rate / len(mini_batch)) * bw / (np.sqrt(cw) + np.finfo(np.float32).eps)
                        for w, bw, cw in zip(self.weights, batch_w, self.cache_w)]
        self.biases = [b - (learn_rate / len(mini_batch)) * bb / (np.sqrt(cb) + np.finfo(np.float32).eps)
                       for b, bb, cb in zip(self.biases, batch_b, self.cache_b)]

    def stochastic(self, train_data, epochs, mini_batch_size, learning_rate, valid_data, test_data):
        n_test = len(test_data)
        n_valid = len(valid_data)
        n = len(train_data)
        for i in xrange(epochs):
            start_time = time.time()
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            for batch in mini_batches:
                self.update(batch, learning_rate)

            self.train_cost.append(self.get_train_cost(train_data))
            self.valid_cost.append(self.get_cost(valid_data))
            self.train_acc.append((float(self.train_test(train_data)) / float(n)) * 100)
            self.valid_acc.append((float(self.test(valid_data)) / float(n_valid)) * 100)
            accuracy = (float(self.test(valid_data)) / float(n_valid)) * 100
            print 'Epoch : {0}'.format(i)
            print 'Accuracy : {0:.2f}%'.format(accuracy)
            """
            print 'Cost : {0:.2f}'.format(error)
            print 'Accuracy : {0:.2f}%'.format(accuracy)
            """
            print 'Epoch Running Time : {0:.2f}'.format(time.time() - start_time)

        error = self.get_cost(test_data)
        accuracy = (float(self.test(test_data)) / float(n_test)) * 100
        print 'Test Data Cost : {0:.2f}'.format(error)
        print 'Test Data Accuracy : {0:.2f}%'.format(accuracy)

        self.save("sigmoid_Adagrad.json", accuracy)

        return self.train_acc, self.valid_acc, self.train_cost, self.valid_cost

    def save(self, filename, accuracy):
        data = {"train_cost": self.train_cost,
                "valid_cost": self.valid_cost,
                "train_accuracy": self.train_acc,
                "valid_accuracy": self.valid_acc,
                "test_accuracy": accuracy,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def dsigmoid(self, z):
        return z * (1.0 - z)

    def _error(self, output_activations, y):
        return output_activations - y

    def train_test(self, test_data):
        results = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)

    def test(self, test_data):
        results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)

    def get_train_cost(self, test_data):
        results = 0.0
        for (x, y) in test_data:
            results += 0.5 * np.linalg.norm(self._error(self.feed_forward(x), y)) ** 2
        results /= float(len(test_data))
        return results

    def get_cost(self, test_data):
        results = 0.0
        for (x, y) in test_data:
            result_labels = self.vectorresult(y)
            results += 0.5 * np.linalg.norm(self._error(self.feed_forward(x), result_labels)) ** 2
        results /= float(len(test_data))
        return results

    def vectorresult(self, j):
        vect = np.zeros((10, 1))
        vect[j] = 1.0
        return vect


if __name__ == '__main__':
    NN = [784, 60, 10]
    LEARNING_RATE = 0.5
    EPOCHS = 100
    MINI_BATCH_SIZE = 10
    TRAIN_DATA = None
    VALID_DATA = None
    TEST_DATA = None

    print 'MNIST Data Loading...'
    data_loader = MNIST_Loader.DataLoader()
    TRAIN_DATA, VALID_DATA, TEST_DATA = data_loader.loaddata()
    print 'MNIST Data Loaded!!!!'

    mlp = MLP()
    total_start_time = time.time()
    train_acc, valid_acc, train_cost, valid_cost = mlp.stochastic(
        TRAIN_DATA,
        EPOCHS,
        MINI_BATCH_SIZE,
        LEARNING_RATE,
        VALID_DATA,
        TEST_DATA
    )
    print 'Training Finished'
    print 'Training Running Time : {0:.2f}'.format(time.time() - total_start_time)
    acc_disp(EPOCHS, train_acc, valid_acc)
    err_disp(EPOCHS, train_cost, valid_cost)


