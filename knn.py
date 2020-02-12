
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import scale

# Загрузите выборку Wine
# Извлеките из данных признаки и классы. Класс записан в первом столбце (три варианта),
# признаки — в столбцах со второго по последний.
data = pd.read_csv('wine.data', names=['class', 'Alcohol', ' Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                                       'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'])

X = data[['Alcohol', ' Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']]
y = data['class']

# Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold).
# Создайте генератор разбиений, который перемешивает выборку перед формированием
# блоков (shuffle=True). Для воспроизводимости результата, создавайте генератор
# KFold с фиксированным параметром random_state=42. В качестве меры качества
# используйте долю верных ответов (accuracy).

kf = KFold(n_splits=5, random_state=42, shuffle=True)
knn = KNeighborsClassifier(n_neighbors=1)

listmean = list()
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    results = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
    listmean.append(np.mean(results))
#listmean.sort(reverse=True)
print(listmean)

'''for train, test in kf:
    X_train, y_train = X[train],y[train]
    X_test, y_test = X[test], y[test]
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)'''

#cross_val_score()

# Найдите точность классификации на кросс-валидации для метода k ближайших соседей
# (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50. При каком k получилось
# оптимальное качество? Чему оно равно (число в интервале от 0 до 1)? Данные
# результаты и будут ответами на вопросы 1 и 2.



with open('knn1.txt', 'w') as f:
    f.write("1")
f.close()
with open('knn2.txt', 'w') as f:
    f.write("0.73")
f.close()


# Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale.
# Снова найдите оптимальное k на кросс-валидации.
X = scale(X)
print(X)

listmean.clear()

for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    results = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
    listmean.append(np.mean(results))

print(listmean.index(0.9776190476190475))

with open('knn3.txt', 'w') as f:
    f.write("28")
f.close()
with open('knn4.txt', 'w') as f:
    f.write("0.98")
f.close()