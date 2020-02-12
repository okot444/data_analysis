import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
# Целевая переменная записана в первом столбце, признаки — во втором и третьем.
dataTrain = pd.read_csv('perceptron-train.csv', names=['y', 'X1', 'X2'])
dataTest = pd.read_csv('perceptron-test.csv', names=['y', 'X1', 'X2'])

y_train = dataTrain['y']
X_train = dataTrain[['X1', 'X2']]

y_test = dataTest['y']
X_test = dataTest[['X1', 'X2']]

print(dataTrain)

# Обучите персептрон со стандартными параметрами и random_state=241.

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


# Подсчитайте качество (долю правильно классифицированных объектов) полученного классификатора на тестовой выборке.
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

# Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучите персептрон на новой выборке. Найдите долю правильных ответов на тестовой выборке.

clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)

accuracy_norm = accuracy_score(y_test, predictions)
print(accuracy_norm)
"sd".format()


with open('perceptron.txt', 'w') as f:
    f.write("{0:.3f}".format(accuracy_norm-accuracy))
f.close()

