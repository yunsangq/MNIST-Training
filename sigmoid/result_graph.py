import matplotlib.pyplot as plt
import json


def err_disp(epochs, sgd, mu, nag, adg, add, rms, adam, ces, l2, l1, dropout, weight, my):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Cost per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), ces, color='#266576', label='CE+Softmax')
    ax.plot(range(epochs), l2, color='#467489', label='L2')
    ax.plot(range(epochs), l1, color='#7d94ab', label='L1')
    ax.plot(range(epochs), dropout, color='#a784b8', label='Dropout')
    ax.plot(range(epochs), weight, color='#70c12e', label='Weight Init')
    ax.plot(range(epochs), my, color='#ff2400', label='SangYun')
    ax.plot(range(epochs), sgd, color='#db5c00', label='SGD')
    ax.plot(range(epochs), mu, color='#d68c1a', label='Momentum')
    ax.plot(range(epochs), nag, color='#cbbb6e', label='NAG')
    ax.plot(range(epochs), adg, color='#99aa3d', label='Adagrad')
    ax.plot(range(epochs), add, color='#519b13', label='Adadelta')
    ax.plot(range(epochs), rms, color='#6d4d9b', label='RMSProp')
    ax.plot(range(epochs), adam, color='#7da7d9', label='Adam')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost(Valid)')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def acc_disp(epochs, sgd, mu, nag, adg, add, rms, adam, ces, l2, l1, dropout, weight, my):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Accuracy per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), ces, color='#266576', label='CE+Softmax')
    ax.plot(range(epochs), l2, color='#467489', label='L2')
    ax.plot(range(epochs), l1, color='#7d94ab', label='L1')
    ax.plot(range(epochs), dropout, color='#a784b8', label='Dropout')
    ax.plot(range(epochs), weight, color='#70c12e', label='Weight Init')
    ax.plot(range(epochs), my, color='#ff2400', label='SangYun')
    ax.plot(range(epochs), sgd, color='#db5c00', label='SGD')
    ax.plot(range(epochs), mu, color='#d68c1a', label='Momentum')
    ax.plot(range(epochs), nag, color='#cbbb6e', label='NAG')
    ax.plot(range(epochs), adg, color='#99aa3d', label='Adagrad')
    ax.plot(range(epochs), add, color='#519b13', label='Adadelta')
    ax.plot(range(epochs), rms, color='#6d4d9b', label='RMSProp')
    ax.plot(range(epochs), adam, color='#7da7d9', label='Adam')

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
    EPOCHS = 50
    
    # sgd, mu, nag, adg, add, rms, adam, ces, l2, l1, dropout, weight, my
    sgd_cost, sgd_accuracy = load("sigmoid_SGD.json")
    mu_cost, mu_accuracy = load("sigmoid_Momentum.json")
    nag_cost, nag_accuracy = load("sigmoid_NAG.json")
    adg_cost, adg_accuracy = load("sigmoid_Adagrad.json")
    add_cost, add_accuracy = load("sigmoid_Adadelta.json")
    rms_cost, rms_accuracy = load("sigmoid_RMSProp.json")
    adam_cost, adam_accuracy = load("sigmoid_Adam.json")

    ces_cost, ces_accuracy = load("sigmoid_CE_softmax.json")
    l2_cost, l2_accuracy = load("sigmoid_l2.json")
    l1_cost, l1_accuracy = load("sigmoid_l1.json")
    dropout_cost, dropout_accuracy = load("sigmoid_CE_softmax_dropout.json")
    weight_cost, weight_accuracy = load("sigmoid_L2_weight.json")
    my_cost, my_accuracy = load("sigmoid_my_1.json")

    acc_disp(EPOCHS, sgd_accuracy, mu_accuracy, nag_accuracy, adg_accuracy, add_accuracy, rms_accuracy, adam_accuracy,
             ces_accuracy, l2_accuracy, l1_accuracy, dropout_accuracy, weight_accuracy, my_accuracy)
    err_disp(EPOCHS, sgd_cost, mu_cost, nag_cost, adg_cost, add_cost, rms_cost, adam_cost,
             ces_cost, l2_cost, l1_cost, dropout_cost, weight_cost, my_cost)

