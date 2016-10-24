import matplotlib.pyplot as plt
import json


def err_disp(epochs, r_sgd, r_mu, r_nag, r_adg, r_add, r_rms, r_adam, s_sgd, s_mu, s_nag, s_adg, s_add, s_rms, s_adam):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Cost per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), r_sgd, color='#FF0000', label='ReLU_SGD')
    ax.plot(range(epochs), r_mu, color='#A0522D', label='ReLU_Momentum')
    ax.plot(range(epochs), r_nag, color='#FF8C00', label='ReLU_NAG')
    ax.plot(range(epochs), r_adg, color='#B8860B', label='ReLU_Adagrad')
    ax.plot(range(epochs), r_add, color='#7FFF00', label='ReLU_Adadelta')
    ax.plot(range(epochs), r_rms, color='#808000', label='ReLU_RMSProp')
    ax.plot(range(epochs), r_adam, color='#556B2F', label='ReLU_Adam')
    ax.plot(range(epochs), s_sgd, color='#008080', label='Sigmoid_SGD')
    ax.plot(range(epochs), s_mu, color='#4682B4', label='Sigmoid_Momentum')
    ax.plot(range(epochs), s_nag, color='#48D1CC', label='Sigmoid_NAG')
    ax.plot(range(epochs), s_adg, color='#4169E1', label='Sigmoid_Adagrad')
    ax.plot(range(epochs), s_add, color='#9932CC', label='Sigmoid_Adadelta')
    ax.plot(range(epochs), s_rms, color='#C71585', label='Sigmoid_RMSProp')
    ax.plot(range(epochs), s_adam, color='#000000', label='Sigmoid_Adam')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost(Valid)')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def acc_disp(epochs, r_sgd, r_mu, r_nag, r_adg, r_add, r_rms, r_adam, s_sgd, s_mu, s_nag, s_adg, s_add, s_rms, s_adam):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Accuracy per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), r_sgd, color='#FF0000', label='ReLU_SGD')
    ax.plot(range(epochs), r_mu, color='#A0522D', label='ReLU_Momentum')
    ax.plot(range(epochs), r_nag, color='#FF8C00', label='ReLU_NAG')
    ax.plot(range(epochs), r_adg, color='#B8860B', label='ReLU_Adagrad')
    ax.plot(range(epochs), r_add, color='#7FFF00', label='ReLU_Adadelta')
    ax.plot(range(epochs), r_rms, color='#808000', label='ReLU_RMSProp')
    ax.plot(range(epochs), r_adam, color='#556B2F', label='ReLU_Adam')
    ax.plot(range(epochs), s_sgd, color='#008080', label='Sigmoid_SGD')
    ax.plot(range(epochs), s_mu, color='#4682B4', label='Sigmoid_Momentum')
    ax.plot(range(epochs), s_nag, color='#48D1CC', label='Sigmoid_NAG')
    ax.plot(range(epochs), s_adg, color='#4169E1', label='Sigmoid_Adagrad')
    ax.plot(range(epochs), s_add, color='#9932CC', label='Sigmoid_Adadelta')
    ax.plot(range(epochs), s_rms, color='#C71585', label='Sigmoid_RMSProp')
    ax.plot(range(epochs), s_adam, color='#000000', label='Sigmoid_Adam')
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
    
    # ReLU sgd, mu, nag, adg, add, rms, adam
    r_sgd_cost, r_sgd_accuracy = load("ReLU_SGD.json")
    r_mu_cost, r_mu_accuracy = load("ReLU_Momentum.json")
    r_nag_cost, r_nag_accuracy = load("ReLU_NAG.json")
    r_adg_cost, r_adg_accuracy = load("ReLU_Adagrad.json")
    r_add_cost, r_add_accuracy = load("ReLU_Adadelta.json")
    r_rms_cost, r_rms_accuracy = load("ReLU_RMSProp.json")
    r_adam_cost, r_adam_accuracy = load("ReLU_Adam.json")

    # sigmoid sgd, mu, nag, adg, add, rms, adam
    s_sgd_cost, s_sgd_accuracy = load("sigmoid_SGD.json")
    s_mu_cost, s_mu_accuracy = load("sigmoid_Momentum.json")
    s_nag_cost, s_nag_accuracy = load("sigmoid_NAG.json")
    s_adg_cost, s_adg_accuracy = load("sigmoid_Adagrad.json")
    s_add_cost, s_add_accuracy = load("sigmoid_Adadelta.json")
    s_rms_cost, s_rms_accuracy = load("sigmoid_RMSProp.json")
    s_adam_cost, s_adam_accuracy = load("sigmoid_Adam.json")

    acc_disp(EPOCHS, r_sgd_accuracy, r_mu_accuracy, r_nag_accuracy, r_adg_accuracy, r_add_accuracy, r_rms_accuracy,
             r_adam_accuracy, s_sgd_accuracy, s_mu_accuracy, s_nag_accuracy, s_adg_accuracy, s_add_accuracy,
             s_rms_accuracy, s_adam_accuracy)
    err_disp(EPOCHS, r_sgd_cost, r_mu_cost, r_nag_cost, r_adg_cost, r_add_cost, r_rms_cost, r_adam_cost, s_sgd_cost,
             s_mu_cost, s_nag_cost, s_adg_cost, s_add_cost, s_rms_cost, s_adam_cost)

