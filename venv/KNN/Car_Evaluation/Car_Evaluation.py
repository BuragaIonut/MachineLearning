import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data')
# print(data.head())

label = preprocessing.LabelEncoder()
buying = label.fit_transform(list(data['buying']))
maint = label.fit_transform(list(data['maint']))
door = label.fit_transform(list(data['door']))
persons = label.fit_transform(list(data['persons']))
lug_boot = label.fit_transform(list(data['lug_boot']))
safety = label.fit_transform(list(data['safety']))
cls = label.fit_transform(list(data['class']))
# print(buying)

predict = 'cls'

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1,random_state = 10)

# print(x_train, y_train)
k = int(input('How many neighbors ??'))
model = KNeighborsClassifier(n_neighbors = k)

model.fit(x_train, y_train)

acc = model.score(x_test,y_test)
print(acc)

predicted = model.predict(x_test)

names = ['unacc', 'acc', 'good', 'very_good']

for x in range(len(x_test)):
    if predicted[x] == y_test[x]:
        print(f'Predicted: {predicted[x]}, Data: {x_test[x]}, Actual: {y_test[x]} Status: {names[y_test[x]]}')
    else:
        print(f'Predicted: {predicted[x]}, Data: {x_test[x]}, Actual: {y_test[x]} Status: {names[y_test[x]]} !!!!!!!!!!!')
print(f'Model accuracy score is {model.score(x_test,y_test) *100:.2f} %')