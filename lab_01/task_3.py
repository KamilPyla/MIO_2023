from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

data = load_iris()

x_data = data.data
y_data = data.target

x_k1 = [ x_data[i] for i in range(len(y_data)) if y_data[i] == 0 ]
x_k2 = [ x_data[i] for i in range(len(y_data)) if y_data[i] == 1 ]
x_k3 = [ x_data[i] for i in range(len(y_data)) if y_data[i] == 2 ]

neuron = Perceptron(tol=1e-3, max_iter = 25)
