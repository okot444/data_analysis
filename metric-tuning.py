
import numpy as np
from numpy import linspace
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
# Результатом вызова данной функции является объект, у которого признаки записаны
# в поле data, а целевой вектор — в поле target.

data = load_boston()
X = data['data']
y = data['target']

X = scale(X)



# Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом,
# чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace).
# Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр
# добавляет в алгоритм веса, зависящие от расстояния до ближайших соседей.
# В качестве метрики качества используйте среднеквадратичную ошибку
# (параметр scoring='mean_squared_error' у cross_val_score; при использовании библиотеки
# scikit-learn версии 0.18.1 и выше необходимо указывать scoring='neg_mean_squared_error').
#  Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам
# с random_state = 42, не забудьте включить перемешивание выборки (shuffle=True).


c = np.linspace(1, 10, 200)


kf = KFold(n_splits=5, random_state=42, shuffle=True)
#knn = KNeighborsRegressor(n_neighbors=5, weights='distance',)

listmean = list()
for i in c:
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=i)
    results = cross_val_score(knn, X, y, cv=kf, scoring='neg_mean_squared_error')
    listmean.append(np.mean(results))
#listmean.sort(reverse=True) -16.030646734221644
print(listmean)
listmean.clear()

# Определите, при каком p качество на кросс-валидации оказалось оптимальным.
# Обратите внимание, что cross_val_score возвращает массив показателей качества
# по блокам; необходимо максимизировать среднее этих показателей.
# Это значение параметра и будет ответом на задачу.


with open('knn5.txt', 'w') as f:
    f.write("1")
f.close()