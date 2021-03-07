import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm

data = pd.read_csv('cmc.data')
# print(data.head())

x = np.array(data.drop(['cm'],axis = 1))
y = np.array(data['cm'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size= 0.33)

model = neighbors.KNeighborsClassifier(n_neighbors= 7)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)
# plt.scatter(x[:,1],x[:,3])
# plt.style.use('fivethirtyeight')
# plt.show()