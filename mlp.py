import random
import numpy as np
# import matplotlib.pyplot as plt
import MNIST
import time

LEARNING_RATE = 3.0
EPOCH = 50
TOTAL_DATA_SIZE = 0
TRAINING_DATA_SIZE = 0
TEST_DATA_SIZE = 0
MINI_BATCH_SIZE = 10
x1 = []
y1 = []

f_x1 = open("./x1.txt", 'w')
f_y1 = open("./y1.txt", 'w')
f_weight_input = open("./weight_input.txt", 'w')
f_weight_output = open("./weight_output.txt", 'w')


def sigmoid(x):
    return 1.0/(1.0+np.exp(-1 * x))


def sigmoid_prime(y):
    return y * (1.0 - y)


class MLP(object):
    def __init__(self, n_input, n_hidden, n_output):
        # + 1 bias
        self.n_input = n_input + 1
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.act_input = [1.0] * self.n_input
        self.act_hidden = [1.0] * self.n_hidden
        self.act_output = [1.0] * self.n_output

        self.weight_input = np.zeros((len(self.act_input), len(self.act_hidden)))
        self.weight_output = np.zeros((len(self.act_hidden), len(self.act_output)))

        self.change_weight_input = np.zeros((len(self.act_input), len(self.act_hidden)))
        self.change_weight_output = np.zeros((len(self.act_hidden), len(self.act_output)))

        for i in range(self.n_input):
            for j in range(self.n_hidden):
                self.weight_input[i][j] = random.uniform(-0.05, 0.05)

        for i in range(self.n_hidden):
            for j in range(self.n_output):
                self.weight_output[i][j] = random.uniform(-0.05, 0.05)

    # RGB 0~255 (x - min) / (max - min)
    def normalize_data(self, x):
        return (x - 0.0) / (255.0 - 0.0)

    def feed_forward(self, inputs):
        for i in range(self.n_input - 1):
            self.act_input[i] = self.normalize_data(inputs[i])
            # self.act_input[i] = inputs[i]

        for j in range(self.n_hidden):
            _sum = 0.0
            for i in range(self.n_input):
                _sum += self.act_input[i] * self.weight_input[i][j]
            self.act_hidden[j] = sigmoid(_sum)

        for k in range(self.n_output):
            _sum = 0.0
            for j in range(self.n_hidden):
                _sum += self.act_hidden[j] * self.weight_output[j][k]
            self.act_output[k] = sigmoid(_sum)
        return self.act_output[:]

    def back_propagate(self, targets):
        output_deltas = [0.0] * self.n_output
        for k in range(self.n_output):
            error = targets[k] - self.act_output[k]
            output_deltas[k] = sigmoid_prime(self.act_output[k]) * error

        hidden_deltas = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            error = 0.0
            for k in range(self.n_output):
                error += output_deltas[k] * self.weight_output[j][k]
            hidden_deltas[j] = sigmoid_prime(self.act_hidden[j]) * error

        for j in range(self.n_hidden):
            for k in range(self.n_output):
                # mini-Batch
                self.change_weight_output[j][k] += output_deltas[k] * self.act_hidden[j]
                # Single-step
                # self.weight_output[j][k] += LEARNING_RATE * output_deltas[k] * self.act_hidden[j]

        for i in range(self.n_input):
            for j in range(self.n_hidden):
                # mini-Batch
                self.change_weight_input[i][j] += hidden_deltas[j] * self.act_input[i]
                # Single-step
                # self.weight_input[i][j] += LEARNING_RATE * hidden_deltas[j] * self.act_input[i]

        error = np.array(targets) - np.array(self.act_output)
        error = np.linalg.norm(error)
        error = 0.5 * error ** 2
        return error

    def update(self):
        for j in range(self.n_hidden):
            for k in range(self.n_output):
                self.weight_output[j][k] += LEARNING_RATE / MINI_BATCH_SIZE * (self.change_weight_output[j][k] / TRAINING_DATA_SIZE)

        for i in range(self.n_input):
            for j in range(self.n_hidden):
                self.weight_input[i][j] += LEARNING_RATE / MINI_BATCH_SIZE * (self.change_weight_input[i][j] / TRAINING_DATA_SIZE)

        self.change_weight_output = np.zeros((len(self.act_hidden), len(self.act_output)))
        self.change_weight_input = np.zeros((len(self.act_input), len(self.act_hidden)))

    def training(self, patterns, outputs):
        for i in range(EPOCH):
            error = 0.0
            mini_batch_idx = np.random.randint(TOTAL_DATA_SIZE, size=TRAINING_DATA_SIZE)
            m_patterns = []
            m_outputs = []
            for idx in mini_batch_idx:
                m_patterns.append(patterns[idx])
                m_outputs.append(outputs[idx])
            start_time = time.time()
            for j in range(TRAINING_DATA_SIZE):
                inputs = m_patterns[j]
                targets = m_outputs[j]
                self.feed_forward(inputs)
                error += self.back_propagate(targets)
                if j % 1000 == 0 and j != 0:
                    tmp = error / float(j)
                    print ('time -> ' + str(j))
                    print ('error -> %-.5f' % tmp)
            print 'Epoch Running Time : %.02f' % (time.time() - start_time)
            error /= float(TRAINING_DATA_SIZE)
            self.update()
            x1.append(i)
            y1.append(error)
            print ('epoch -> ' + str(i))
            print ('epoch total error -> %-.5f' % error)

    def test(self, patterns, labels):
        error = 0.0
        correct = 0.0
        for i in range(TEST_DATA_SIZE):
            output = self.feed_forward(patterns[i])
            tmp = np.array(labels[i]) - np.array(self.act_output)
            tmp = np.linalg.norm(tmp)
            error += 0.5 * tmp ** 2
            t = np.array(labels[i]).argmax()
            o = np.array(output).argmax()
            if t == o:
                correct += 1.0
            # print(t, '->', o)
            # print (patterns[i], output)
        error /= float(TEST_DATA_SIZE)
        print ('TEST error : %-.5f' % error)
        acc = correct / TEST_DATA_SIZE * 100
        print ('Accuracy : ' + str(acc))

        for j in range(self.n_hidden):
            for k in range(self.n_output):
                f_weight_output_data = "%f," % self.weight_output[j][k]
                f_weight_output.write(f_weight_output_data)
            f_weight_output_data = "\n"
            f_weight_output.write(f_weight_output_data)

        for i in range(self.n_input):
            for j in range(self.n_hidden):
                f_weight_input_data = "%f," % self.weight_input[i][j]
                f_weight_input.write(f_weight_input_data)
            f_weight_input_data = "\n"
            f_weight_input.write(f_weight_input_data)

if __name__ == '__main__':
    mn = MNIST.MNIST()
    img, label = mn.load_training()
    TOTAL_DATA_SIZE = len(img)
    TRAINING_DATA_SIZE =  TOTAL_DATA_SIZE / MINI_BATCH_SIZE
    # TRAINING_DATA_SIZE = 10
    test_img, test_label = mn.load_testing()
    TEST_DATA_SIZE = len(test_img)

    NN = MLP(784, 60, 10)

    total_start_time = time.time()
    NN.training(img, label)
    print 'Training Running Time : %.02f' % (time.time() - total_start_time)

    NN.test(test_img, test_label)

    for x in x1:
        x1_data = "%d\n" % x
        f_x1.write(x1_data)

    for y in y1:
        y1_data = "%f\n" % y
        f_y1.write(y1_data)

    f_x1.close()
    f_y1.close()
    f_weight_input.close()
    f_weight_output.close()
"""
    plt.plot(x1, y1)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.show()
"""
"""
   img = [
       [0, 0],
       [0, 1],
       [1, 0],
       [1, 1]
   ]
   label = [
       [0],
       [1],
       [1],
       [0]
   ]
   test_img = [
       [0, 0],
       [0, 1],
       [1, 0],
       [1, 1]
   ]
   test_label = [
       [0],
       [1],
       [1],
       [0]
   ]
   TRAINING_DATA_SIZE = len(img)
   TEST_DATA_SIZE = len(test_img)

   NN = MLP(2, 4, 1)
   """
