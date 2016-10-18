import random
import numpy as np
import MNIST_Loader
import matplotlib.pyplot as plt
import time


def disp(epochs, hist):
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
        output_delta = self.cost(activations[-1], y) * self.dsigmoid(activations[-1])
        delta_b[-1] = output_delta
        delta_w[-1] = np.dot(output_delta, activations[-2].transpose())

        for i in xrange(2, 3):
            hidden_delta = np.dot(self.weights[-i+1].transpose(), output_delta) * self.dsigmoid(activations[-i])
            delta_b[-i] = hidden_delta
            delta_w[-i] = np.dot(hidden_delta, activations[-i-1].transpose())
        return delta_b, delta_w

    def update(self, mini_batch, learn_rate):
        batch_b = [np.zeros(b.shape) for b in self.biases]
        batch_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_batch_b, delta_batch_w = self.train(x, y)
            batch_b = [b + delta_b for b, delta_b in zip(batch_b, delta_batch_b)]
            batch_w = [w + delta_w for w, delta_w in zip(batch_w, delta_batch_w)]

        self.weights = [w - (learn_rate / len(mini_batch)) * bw for w, bw in zip(self.weights, batch_w)]
        self.biases = [b - (learn_rate / len(mini_batch)) * bb for b, bb in zip(self.biases, batch_b)]

    def stochastic(self, train_data, epochs, mini_batch_size, learn_rate, test_data):
        n_test = len(test_data)
        n = len(train_data)
        for i in xrange(epochs):
            start_time = time.time()
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            for batch in mini_batches:
                self.update(batch, learn_rate)

            accuracy = (float(self.test(test_data)) / float(n_test)) * 100
            self.epoch_accuracy.append(accuracy)
            print 'Epoch : {0}'.format(i)
            print 'Accuracy : {0:.2f}%'.format(accuracy)
            print 'Epoch Running Time : %.02f' % (time.time() - start_time)
        return self.epoch_accuracy

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def dsigmoid(self, z):
        return z * (1.0 - z)

    def cost(self, output_activations, y):
        return output_activations - y

    def test(self, test_data):
        results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)


if __name__ == '__main__':
    NN = [784, 60, 10]
    LEARN_RATE = 0.5
    EPOCHS = 50
    MINI_BATCH_SIZE = 10
    TRAIN_DATA = None
    TEST_DATA = None

    print 'MNIST Data Loading...'
    data_loader = MNIST_Loader.DataLoader()
    TRAIN_DATA, TEST_DATA = data_loader.LoadData()
    print 'MNIST Data Loaded!!'

    mlp = MLP()
    total_start_time = time.time()
    epoch_accuracy = mlp.stochastic(
        TRAIN_DATA,
        EPOCHS,
        MINI_BATCH_SIZE,
        LEARN_RATE,
        TEST_DATA
    )
    print 'Training Finished'
    print 'Training Running Time : %.02f' % (time.time() - total_start_time)
    disp(EPOCHS, epoch_accuracy)



