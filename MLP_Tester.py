import random
import numpy as np
import MNIST_Loader
import matplotlib.pyplot as plt
import time
import json


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

if __name__ == '__main__':
    NN = [784, 60, 10]
    LEARNING_RATE = 0.5
    EPOCHS = 50
    MINI_BATCH_SIZE = 10
    TRAIN_DATA = None
    VALID_DATA = None
    TEST_DATA = None

    print 'MNIST Data Loading...'
    data_loader = MNIST_Loader.DataLoader()
    TRAIN_DATA, VALID_DATA, TEST_DATA = data_loader.loaddata()
    print 'MNIST Data Loaded!!!!'



    # acc_disp(EPOCHS, epoch_accuracy)
    # err_disp(EPOCHS, epoch_cost)

