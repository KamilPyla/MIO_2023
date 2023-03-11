import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix


def split_data(ratio, d):
    data_len = len(d)
    part_1 = int(ratio*data_len)
    return d[:part_1], d[part_1:]

data = load_iris()

x_data = data.data
y_data = data.target

x_k1 = [ x_data[i] for i in range(len(y_data)) if y_data[i] == 0 ]
x_k2 = [ x_data[i] for i in range(len(y_data)) if y_data[i] == 1 ]
x_k3 = [ x_data[i] for i in range(len(y_data)) if y_data[i] == 2 ]


for _ in range(5):
    random.shuffle(x_k1)
    random.shuffle(x_k2)
    random.shuffle(x_k3)

    k1_learn, k1_test = split_data(0.8, x_k1)
    k2_learn, k2_test = split_data(0.8, x_k2)
    k3_learn, k3_test = split_data(0.8, x_k3)

    for i in range(5):
        x_learn = np.concatenate((k1_learn, k2_learn, k3_learn))
        y_learn = np.concatenate((np.array([0]*len(k1_learn)), np.array([1]*len(k2_learn)), np.array([2]*len(k1_learn))))

        x_test = np.concatenate((k1_test, k2_test, k3_test))
        y_test = np.concatenate((np.array([0]*len(k1_test)), np.array([1]*len(k2_test)), np.array([2]*len(k1_test))))
        neuron = Perceptron(tol=1e-3, max_iter = 25)
        neuron.fit(x_learn, y_learn)
        neuron.score(x_test, y_test)

        print(f'Numer interacji: {i}, dokładność: {neuron.score(x_test, y_test)}')

        y_predicted = neuron.predict(x_test)
        confusion_matrix_model = confusion_matrix(y_test, y_predicted)
        print(confusion_matrix_model)
