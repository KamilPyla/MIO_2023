from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('fuel.txt')
x = data[:,0:3]
y = data[:,-1]

for i in range(5):
  neuron = Perceptron(tol=1e-3, max_iter = 25)
  neuron.fit(x, y)
  neuron.score(x, y)
  print(f'Numer interacji: {i}, dokładność: {neuron.score(x, y)}')

  y_predicted = neuron.predict(x)
  confusion_matrix_model = confusion_matrix(y, y_predicted)
  print(confusion_matrix_model)

  

