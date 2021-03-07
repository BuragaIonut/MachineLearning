import numpy as np
import pandas as pd
import sklearn
from  sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv('breast-cancer-wisconsin.data')

# print(data.head())

label = preprocessing.LabelEncoder()
CT = label.fit_transform(list(data['CT']))
SZ = label.fit_transform(list(data['SZ']))
SH = label.fit_transform(list(data['SH']))
MA = label.fit_transform(list(data['MA']))
SECS = label.fit_transform(list(data['SECS']))
BN = label.fit_transform(list(data['BN']))
NN = label.fit_transform(list(data['NN']))
M = label.fit_transform(list(data['M']))
CLS = label.fit_transform(list(data['CLS']))




# predict = 'CLS'
x = list(zip(CT, SZ, SH, MA,SECS,BN,NN,M))

y = list(CLS)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size= 0.1)

model = KNeighborsClassifier(n_neighbors = 7)

model.fit(x_train,y_train)
print(f'The model has an accuracy of :{model.score(x_test,y_test)*100:.2f} %')

predicted = model.predict(x_test)

cls = ['benign','malign']

for i in range(len(predicted)):
    print(f'Predicted: {cls[predicted[i]]} Actual: {cls[y_test[i]]}') if predicted[i] == y_test[i] else print(f'Predicted: {cls[predicted[i]]} Actual: {cls[y_test[i]]} !!!!!!!')