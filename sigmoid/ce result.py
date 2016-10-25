import matplotlib.pyplot as plt
import json


def err_disp(epochs, sgd, mu, nag):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Cost per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), sgd, color='#FF0000', label='SGD')
    ax.plot(range(epochs), mu, color='#A0522D', label='Cross-entropy')
    ax.plot(range(epochs), nag, color='#FF8C00', label='CE+Softmax')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost(Valid)')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def acc_disp(epochs, sgd, mu, nag):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Accuracy per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), sgd, color='#FF0000', label='SGD')
    ax.plot(range(epochs), mu, color='#A0522D', label='Cross-entropy')
    ax.plot(range(epochs), nag, color='#FF8C00', label='CE+Softmax')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy(Valid)')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()

    """
    data = {"train_cost": self.train_cost,
                "valid_cost": self.valid_cost,
                "train_accuracy": self.train_acc,
                "valid_accuracy": self.valid_acc,
                "test_accuracy": accuracy,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
    """
    cost = data["train_cost"]
    accuracy = data["train_accuracy"]

    return cost, accuracy


if __name__ == '__main__':
    EPOCHS = 50

    # sgd, mu, nag, adg, add, rms, adam
    sgd_cost, sgd_accuracy = load("sigmoid_SGD.json")
    mu_cost, mu_accuracy = load("sigmoid_cross-entropy_only.json")
    nag_cost, nag_accuracy = load("sigmoid_cross-entropy.json")

    acc_disp(EPOCHS, sgd_accuracy, mu_accuracy, nag_accuracy)
    err_disp(EPOCHS, sgd_cost, mu_cost, nag_cost)

