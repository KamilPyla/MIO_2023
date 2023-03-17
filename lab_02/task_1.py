import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = np.genfromtxt('./treatment.txt', delimiter=',')
y = data[:,2].astype(int)
x = data[:,:2]
max_value1 = x[:,1].max()
min_value1 = x[:,1].min()
devider1 = max_value1 - min_value1

max_value0 = x[:,0].max()
min_value0 = x[:,0].min()
devider0 = max_value0 - min_value0

x[:,0] = x[:,0] / devider0
x[:,1] = x[:,1] / devider1

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size=0.2)

model = MLPClassifier(hidden_layer_sizes = (50,50), max_iter=2000)
model.fit(x_train, y_train)
print('score:', model.score(x_test,y_test))

y_predicted = model.predict(x_test)

plt.scatter(x_train[:,0], x_train[:,1], c=['k' if i == 0 else 'r' for i in y_train])
plt.scatter(x_test[:,0], x_test[:,1], c=['g' if i == 0 else 'b' for i in y_predicted])

# y_predicted

plt.scatter(x[:,0], x[:,1], c=['k' if i == 0 else 'r' for i in y])
xx, yy = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
test_points = np.transpose(np.vstack((np.ravel(xx),np.ravel(yy))))

prediction = model.predict(test_points)
plt.scatter(test_points[:,0], test_points[:,1], c=prediction)
