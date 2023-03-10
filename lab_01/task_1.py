from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

x_test = np.concatenate((np.random.normal([0,-1],[1,1],[200,2]), np.random.normal([1,1],[1,1],[200,2])))
y_test = np.concatenate((np.array([0]*200),np.array([1]*200)))

counter = [5, 10, 20, 100]

for i in counter:
  x_learn = np.concatenate((np.random.normal([0,-1],[1,1],[i,2]), np.random.normal([1,1],[1,1],[i,2])))
  y_learn = np.concatenate((np.array([0]*i),np.array([1]*i)))
  neuron = Perceptron(tol=1e-3, max_iter = 25)
  neuron.fit(x_learn, y_learn)
  neuron.score(x_test, y_test)

  print(f'Dokładność dla {i} ilosci prob (dane testujące): {neuron.score(x_test, y_test)}')
  print(f'Dokładność dla {i} ilosci prob (dane uczące) : {neuron.score(x_learn, y_learn)}')
  x1 = np.linspace(-8, 8, 200)
  x2 = -(1./neuron.coef_[0][1])*(neuron.coef_[0][0]*x1+neuron.intercept_[0])

  plt.plot(x1, x2, '-c')
  plt.xlim(-8,8)
  plt.ylim(-8,8)
  plt.scatter(x_test[:,0], x_test[:,1], c= ['g' if i==0 else 'b' for i in y_test])
  plt.scatter(x_learn[:,0], x_learn[:,1], c= ['r' if i==0  else 'k' for i in y_learn])
  plt.show()
