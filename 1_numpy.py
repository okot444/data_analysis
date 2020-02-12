import numpy
matrix = numpy.random.normal(loc=1, scale=10, size=(1000, 50))

average = numpy.mean(matrix, axis=0)
std = numpy.std(matrix, axis=0)

mx = (matrix - average)/std

# операции над матрицами
Z = numpy.array([[4, 5, 0],
             [1, 9, 3],
             [5, 1, 1],
             [3, 3, 3],
             [9, 9, 9],
             [4, 7, 1]])

x = numpy.sum(Z, axis=1)
print(numpy.nonzero(x > 10))

# Объединение матриц

m1 = numpy.eye(3)
m2 = numpy.eye(3)
print(numpy.vstack((m1, m2)))



