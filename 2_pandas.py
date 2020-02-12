import pandas
import collections

data = pandas.read_csv('titanic.csv', index_col='PassengerId')  # колонка PassengerId задает нумерацию строк данного

# датафрейма


# Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.

print(len(data.loc[data['Sex'] == 'male']), end=" ")
print(len(data.loc[data['Sex'] == 'female']))

with open('1.txt', 'w') as f:
  f.write(str(len(data.loc[data['Sex'] == 'male'])) + ' ' + str(len(data.loc[data['Sex'] == 'female'])))
f.close()

# Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.

pas_num = len(data)
# print(pas_num)
print(len((data[lambda x: x['Survived'] == 1])), end=" ")

print("{:0.2f}".format(len(data.loc[data['Survived'] == 1]) / pas_num * 100))  # округление (НЕ ОТСЕЧЕНИЕ избыточных разрядов)

f = open('2.txt', 'w')
f.write("{:0.2f}".format(len(data.loc[data['Survived'] == 1]) / pas_num * 100))
f.close()

# Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.

print("{:0.2f}".format(len(data.loc[data['Pclass'] == 1]) / pas_num * 100))

f = open('3.txt', 'w')
f.write("{:0.2f}".format(len(data.loc[data['Pclass'] == 1]) / pas_num * 100))
f.close()

# Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел.

print("{:0.2f}".format(data['Age'].mean()), end=" ")
print(data['Age'].median())

f = open('4.txt', 'w')
f.write("{:0.2f}".format(data['Age'].mean()) + ' ' + "{:0.2f}".format(data['Age'].median()))
f.close()

# Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.

# print(data[['SibSp', 'Parch']])
# ?? разобраться

print("{:0.2f}".format(data['SibSp'].corr(data['Parch'])))

f = open('5.txt', 'w')
f.write("{:0.2f}".format(data['SibSp'].corr(data['Parch'])))
f.close()

# Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name)
# его личное имя (First Name). Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
# Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию.
# Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен,
# а также разделения их на женские и мужские.


tfemale = data[lambda x: x['Sex'] == 'female']
fnames = collections.Counter()
for i in tfemale['Name']:
    ind = i.find('(')
    if ind != -1 :
        name = i[ind+1:-1].split(' ')
        #print(name[0])
        fnames[name[0]] += 1
    else:
        ind = i.find('.')
        name = i[ind+2:].split(' ')
        #print(name[0])
        fnames[name[0]] += 1

#print(fnames.most_common(1)[0][0])

f = open('6.txt', 'w')
f.write(fnames.most_common(1)[0][0])
f.close()


print (data)

