import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd


data = datasets.load_breast_cancer()

X_train , X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2)

cls = svm.SVC(kernel ='linear')

cls.fit(X_train, y_train)

pred = cls.predict(X_test)

print('accuracy:', metrics.accuracy_score(y_test, y_pred=pred))
print('precision:', metrics.precision_score(y_test, y_pred=pred))
