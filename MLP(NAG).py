import random
import numpy as np
import MNIST_Loader
import matplotlib.pyplot as plt
import time


def err_disp(epochs, hist):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Cost per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), hist, color='#1F77B4')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost')
    plt.show()


def acc_disp(epochs, hist):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Accuracy per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), hist, color='#1F77B4')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    plt.show()


class MLP:
    def __init__(self):
        self.biases = [np.random.randn(y, 1) for y in NN[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(NN[:-1], NN[1:])]
        self.epoch_accuracy = []
        self.cost = []
        self.before_b = [np.zeros(b.shape) for b in self.biases]
        self.before_w = [np.zeros(w.shape) for w in self.weights]
        self.prev_vb = [np.zeros(b.shape) for b in self.biases]
        self.prev_vw = [np.zeros(w.shape) for w in self.weights]

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
        # outdelta
        delta = self._error(activations[-1], y) * self.dsigmoid(activations[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].transpose())

        for i in xrange(2, 3):
            # hiddendelta
            delta = np.dot(self.weights[-i+1].transpose(), delta) * self.dsigmoid(activations[-i])
            delta_b[-i] = delta
            delta_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return delta_b, delta_w

    def update(self, mini_batch, learn_rate, momentum):
        batch_b = [np.zeros(b.shape) for b in self.biases]
        batch_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_batch_b, delta_batch_w = self.train(x, y)
            batch_b = [b + delta_b for b, delta_b in zip(batch_b, delta_batch_b)]
            batch_w = [w + delta_w for w, delta_w in zip(batch_w, delta_batch_w)]

        """
        vprev = v
        v = -lr*dx + mu*v
        x += -mu*-lr*dx + mu*vprev + (1+mu)*-lr*dx + mu*v
        """

        self.weights = [w - momentum * (-(learn_rate / len(mini_batch)) * bw + momentum * pvw)
                        + (1 + momentum) * (-(learn_rate / len(mini_batch)) * bw + momentum * bew)
                        for w, bw, bew, pvw in zip(self.weights, batch_w, self.before_w, self.prev_vw)]
        self.prev_vw = self.before_w
        self.before_w = [-(learn_rate / len(mini_batch)) * bw + momentum * bew
                         for bw, bew in zip(batch_w, self.before_w)]

        self.biases = [b - momentum * (-(learn_rate / len(mini_batch)) * bb + momentum * pvb)
                       + (1 + momentum) * (-(learn_rate / len(mini_batch)) * bb + momentum * beb)
                       for b, bb, beb, pvb in zip(self.biases, batch_b, self.before_b, self.prev_vb)]
        self.prev_vb = self.before_b
        self.before_b = [-(learn_rate / len(mini_batch)) * bb + momentum * beb
                         for bb, beb in zip(batch_b, self.before_b)]

    def stochastic(self, train_data, epochs, mini_batch_size, learning_rate, valid_data, test_data, momentum):
        n_test = len(test_data)
        n_valid = len(valid_data)
        n = len(train_data)
        for i in xrange(epochs):
            start_time = time.time()
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            for batch in mini_batches:
                self.update(batch, learning_rate, momentum)

            error = self.get_cost(valid_data)
            self.cost.append(error)
            accuracy = (float(self.test(valid_data)) / float(n_valid)) * 100
            self.epoch_accuracy.append(accuracy)
            print 'Epoch : {0}'.format(i)
            print 'Cost : {0:.2f}'.format(error)
            print 'Accuracy : {0:.2f}%'.format(accuracy)
            print 'Epoch Running Time : %.02f' % (time.time() - start_time)

        error = self.get_cost(test_data)
        accuracy = (float(self.test(test_data)) / float(n_test)) * 100
        print 'Test Data Cost : {0:.2f}'.format(error)
        print 'Test Data Accuracy : {0:.2f}%'.format(accuracy)

        return self.epoch_accuracy, self.cost

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def dsigmoid(self, z):
        return z * (1.0 - z)

    def _error(self, output_activations, y):
        return output_activations - y

    def test(self, test_data):
        results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)

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
    MOMENTUM = 0.5

    print 'MNIST Data Loading...'
    data_loader = MNIST_Loader.DataLoader()
    TRAIN_DATA, VALID_DATA, TEST_DATA = data_loader.loaddata()
    print 'MNIST Data Loaded!!!!'

    mlp = MLP()
    total_start_time = time.time()
    epoch_accuracy, epoch_cost = mlp.stochastic(
        TRAIN_DATA,
        EPOCHS,
        MINI_BATCH_SIZE,
        LEARNING_RATE,
        VALID_DATA,
        TEST_DATA,
        MOMENTUM
    )
    print 'Training Finished'
    print 'Training Running Time : %.02f' % (time.time() - total_start_time)
    acc_disp(EPOCHS, epoch_accuracy)
    err_disp(EPOCHS, epoch_cost)


