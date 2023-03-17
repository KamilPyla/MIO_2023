import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

digits = load_digits()
x = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size=0.2)

model = MLPClassifier(hidden_layer_sizes = (13,13), max_iter=2000)
model.fit(x_train, y_train)
print('score:', model.score(x_test,y_test))
