import random
import MNIST
import numpy as np
import matplotlib.pyplot as plt

EPOCH = 10000
TRAINING_DATA = 0
TEST_DATA = 0
x1 = []
y1 = []

class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, n_inputs, n_hidden, n_outputs, weights_hidden=None, bias_hidden=None,
                 weights_output=None, bias_output=None):
        self.n_inputs = n_inputs

        self.hidden_layer = NeuronLayer(n_hidden, bias_hidden)
        self.output_layer = NeuronLayer(n_outputs, bias_output)

        self.init_weights_inputs_to_hidden(weights_hidden)
        self.init_weights_hidden_to_output(weights_output)

        self.change_weights_inputs_to_hidden = np.zeros((len(self.hidden_layer.neurons), self.n_inputs))
        self.change_weights_hidden_to_output = np.zeros((len(self.output_layer.neurons), len(self.hidden_layer.neurons)))

    def init_weights_inputs_to_hidden(self, weights_hidden):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.n_inputs):
                if not weights_hidden:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(weights_hidden[weight_num])
                weight_num += 1

    def init_weights_hidden_to_output(self, weights_output):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not weights_output:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(weights_output[weight_num])
                weight_num += 1

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        output_deltas = [0.0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            output_deltas[o] = self.output_layer.neurons[o].calculate_delta(training_outputs[o])

        hidden_deltas = [0.0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            error = 0.0
            for o in range(len(self.output_layer.neurons)):
                error += output_deltas[o] * self.output_layer.neurons[o].weights[h]
            hidden_deltas[h] = error * self.hidden_layer.neurons[h].sigmoid_prime()

        for o in range(len(self.output_layer.neurons)):
            for w in range(len(self.output_layer.neurons[o].weights)):
                change = self.LEARNING_RATE * output_deltas[o] \
                         * self.output_layer.neurons[o].calculate_total_input_weight(w)
                self.change_weights_hidden_to_output[o][w] += change

        for h in range(len(self.hidden_layer.neurons)):
            for w in range(len(self.hidden_layer.neurons[h].weights)):
                change = self.LEARNING_RATE * hidden_deltas[h] \
                         * self.hidden_layer.neurons[o].calculate_total_input_weight(w)
                self.change_weights_inputs_to_hidden[h][w] += change

    def update(self):
        for o in range(len(self.output_layer.neurons)):
            for w in range(len(self.output_layer.neurons[o].weights)):
                self.output_layer.neurons[o].weights[w] -= self.change_weights_hidden_to_output[o][w] / TRAINING_DATA
                # self.output_layer.neurons[o].weights[w] -= self.change_weights_hidden_to_output[o][w] / 60000.0

        for h in range(len(self.hidden_layer.neurons)):
            for w in range(len(self.hidden_layer.neurons[h].weights)):
                self.hidden_layer.neurons[h].weights[w] -= self.change_weights_inputs_to_hidden[h][w] / TRAINING_DATA
                # self.hidden_layer.neurons[h].weights[w] -= self.change_weights_inputs_to_hidden[h][w] / 60000.0

    def calculate_total_error(self, training_sets):
        total_error = 0.0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

    def test(self, test_sets):
        for t in range(len(test_sets)):
            test_inputs, test_outputs = test_sets[t]
            self.feed_forward(test_inputs)
            for o in range(len(test_outputs)):
                print(test_sets[t][1][o], self.output_layer.neurons[o].output)


class NeuronLayer:
    def __init__(self, n_neurons, bias):
        if bias:
            self.bias = bias
        else:
            self.bias = random.random()

        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(self.bias))

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.inputs = []
        self.output = 0.0

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0.0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    def calculate_delta(self, target_output):
        return self.calculate_error_output(target_output) * self.sigmoid_prime()

    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def calculate_error_output(self, target_output):
        return -(target_output - self.output)

    def sigmoid_prime(self):
        return self.output * (1.0 - self.output)

    def calculate_total_input_weight(self, index):
        return self.inputs[index]


if __name__ == '__main__':
    # mn = MNIST.MNIST()
    # img, label = mn.load_training()
    # test_img, test_label = mn.load_testing()

    """
    nn = NeuralNetwork(2, 2, 2, [0.15, 0.2, 0.25, 0.3], 0.35,
                       [0.4, 0.45, 0.5, 0.55], 0.6)
    for i in range(10000):
        nn.train([0.05, 0.1], [0.01, 0.99])
        nn.update()
        print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))
    """
    # XOR example:

    training_sets = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
    TRAINING_DATA = len(training_sets)
    for i in range(EPOCH):
        for j in range(TRAINING_DATA):
            training_inputs, training_outputs = random.choice(training_sets)
            nn.train(training_inputs, training_outputs)
        nn.update()
        error = nn.calculate_total_error(training_sets)
        x1.append(i)
        y1.append(error)
        print(i+1, error)

    nn.test(training_sets)
    plt.plot(x1, y1)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    # plt.show()
