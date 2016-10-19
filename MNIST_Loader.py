import gzip
import cPickle
import numpy as np


class DataLoader:
    def __init__(self):
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            self.train_set, self.valid_set, self.test_set = cPickle.load(f)

    def loaddata(self):
        train_input = [np.reshape(x, (784, 1)) for x in self.train_set[0]]
        train_result = [self.vectorresult(y) for y in self.train_set[1]]
        train_data = zip(train_input, train_result)

        valid_input = [np.reshape(x, (784, 1)) for x in self.valid_set[0]]
        valid_data = zip(valid_input, self.valid_set[1])

        test_input = [np.reshape(x, (784, 1)) for x in self.test_set[0]]
        test_data = zip(test_input, self.test_set[1])
        return train_data, valid_data, test_data

    def vectorresult(self, j):
        vect = np.zeros((10, 1))
        vect[j] = 1.0
        return vect