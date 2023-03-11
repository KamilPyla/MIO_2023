import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix

data = np.genfromtxt('./fuel.txt')
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

  print(f'Współczynniki {neuron.coef_}')

  x1 = np.linspace(-3, 3, 200)
  x2 = np.linspace(-3, 3, 200)
  x1, x2 = np.meshgrid(x1, x2)
  x3 = -(1./neuron.coef_[0][2])*(neuron.coef_[0][0]*x1 + neuron.coef_[0][1]*x2 + neuron.intercept_[0])

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(x[:,0], x[:,1], x[:,2], c=[1 if cl == 1.0 else 0 for cl in y ])
  ax.plot_surface(x1, x2, x3)
  plt.show()

