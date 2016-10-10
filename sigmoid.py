import random
import numpy as np


def sigmoid(x):
    return 1.0/(1+np.exp(-x))

class neuron(object):
    def __init__(self, num, alpha):
        self.num = num
        self.input_weight = []
        self.alpha = alpha
        self.weight_error = []

    def random_weight(self):
        for i in range(self.num + 1):
            self.input_weight.append(random.uniform(-1, 1))
            self.weight_error.append(0.0)

    def work(self, _input):
        _sum = 0
        for i in range(self.num):
            _sum += _input[i] * self.input_weight[i]

        _sum += self.input_weight[self.num] * 1.0

        return sigmoid(_sum)

    def learn(self, _input, target):
        output = self.work(_input)
        output_error = output - target
        for i in range(self.num):
            self.weight_error[i] += output_error * _input[i] * output * (1-output)
        self.weight_error[self.num] += output_error * 1.0 * output * (1-output)

    def fix(self):
        for i in range(self.num + 1):
            self.input_weight[i] -= self.alpha * self.weight_error[i]
            self.weight_error[i] = 0.0

and_neuron = neuron(2, 0.1)
and_neuron.random_weight()

sample_input = {
    0: [0, 0],
    1: [0, 1],
    2: [1, 0],
    3: [1, 1]
}
sample_output = [0, 0, 0, 1]

for i in range(10000):
    for j in range(4):
        and_neuron.learn(sample_input[j], sample_output[j])
    and_neuron.fix()

    if (i+1)%100 == 0:
        print "---------Learn " + str(i+1) + " times---------"
        for k in range(4):
            print str(sample_input[k][0]) + " " + str(sample_input[k][1]) + " : " + str(and_neuron.work(sample_input[k]))
