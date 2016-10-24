import matplotlib.pyplot as plt
import json


def err_disp(epochs, sgd, mu, nag, adg, add, rms, adam):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Cost per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), sgd, color='#1F77B4', label='SGD')
    ax.plot(range(epochs), mu, color='#b41f1f', label='Momentum')
    ax.plot(range(epochs), nag, color='#fff71e', label='NAG')
    ax.plot(range(epochs), adg, color='#bfff1e', label='Adagrad')
    ax.plot(range(epochs), add, color='#1eff74', label='Adadelta')
    ax.plot(range(epochs), rms, color='#ffc61e', label='RMSProp')
    ax.plot(range(epochs), adam, color='#ff1eec', label='Adam')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost(Valid)')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def acc_disp(epochs, sgd, mu, nag, adg, add, rms, adam):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Accuracy per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), sgd, color='#1F77B4', label='SGD')
    ax.plot(range(epochs), mu, color='#b41f1f', label='Momentum')
    ax.plot(range(epochs), nag, color='#fff71e', label='NAG')
    ax.plot(range(epochs), adg, color='#bfff1e', label='Adagrad')
    ax.plot(range(epochs), add, color='#1eff74', label='Adadelta')
    ax.plot(range(epochs), rms, color='#ffc61e', label='RMSProp')
    ax.plot(range(epochs), adam, color='#ff1eec', label='Adam')
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
    cost = data["valid_cost"]
    accuracy = data["valid_accuracy"]

    return cost, accuracy


if __name__ == '__main__':
    EPOCHS = 100
    
    # sgd, mu, nag, adg, add, rms, adam
    sgd_cost, sgd_accuracy = load("ReLU_SGD.json")
    mu_cost, mu_accuracy = load("ReLU_Momentum.json")
    nag_cost, nag_accuracy = load("ReLU_NAG.json")
    adg_cost, adg_accuracy = load("ReLU_Adagrad.json")
    add_cost, add_accuracy = load("ReLU_Adadelta.json")
    rms_cost, rms_accuracy = load("ReLU_RMSProp.json")
    adam_cost, adam_accuracy = load("ReLU_Adam.json")

    acc_disp(EPOCHS, sgd_accuracy, mu_accuracy, nag_accuracy, adg_accuracy, add_accuracy, rms_accuracy, adam_accuracy)
    err_disp(EPOCHS, sgd_cost, mu_cost, nag_cost, adg_cost, add_cost, rms_cost, adam_cost)

