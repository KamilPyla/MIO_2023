from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

digits = load_digits()
x = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size=0.2)

solvers = ['adam'] # ['sgd', 'lbfgs', 'adam']
activations = ['identity', 'logistic', 'tanh', 'relu']
learning_rates = ['constant', 'invscaling', 'adaptive']

for s in solvers:
  for a in activations:
    for l in learning_rates:

      model = MLPClassifier(solver=s, activation=a, learning_rate=l, hidden_layer_sizes = (13, 13), max_iter=2000)
      model.fit(x_train, y_train)

      predicted = model.predict(x_train)
      matrix = confusion_matrix(y_train, predicted)
      
      print(f'solver: {s}')
      print(f'activation: {a}')
      print(f'learning_rate: {l}')
      print('score:', model.score(x_test,y_test))
      print(f'confusion_matrix: {matrix}')
