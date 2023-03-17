import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = np.loadtxt('yeast.data', usecols=[1,2,3,4,5,6,7,8,9])
y = data[:,8].astype(int)
x = data[:,:8]

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size=0.2)

solvers = ['adam'] # ['sgd', 'lbfgs', 'adam']
activations = ['tanh'] #['identity', 'logistic', 'tanh', 'relu']
learning_rates = ['adaptive'] #['constant', 'invscaling', 'adaptive']

for s in solvers:
  for a in activations:
    for l in learning_rates:

      model = MLPClassifier(solver=s, activation=a, learning_rate=l, hidden_layer_sizes = (13, 13), max_iter=2000)
      model.fit(x_train, y_train)

      train_predicted = model.predict(x_train)
      test_predicted = model.predict(x_test)
      matrix_train = confusion_matrix(y_train, train_predicted)
      matrix_test = confusion_matrix(y_test, test_predicted)
      
      print(f'solver: {s}')
      print(f'activation: {a}')
      print(f'learning_rate: {l}')
      print('score:', model.score(x_test,y_test))
      print(f'confusion_matrix train: {matrix_train}')
      print(f'confusion_matrix test: {matrix_test}')
