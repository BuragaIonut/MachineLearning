import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Fish.csv')
# print(data.head())
data = data[['Weight','Length1','Length2','Length3','Height','Width']]
# print(data.head())

predict = 'Weight'
X = np.array(data.drop(['Weight'],1))
y = np.array(data[predict])

X_train , X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

linear = LinearRegression()

linear.fit(X_train, y_train)
accuracy = linear.score(X_test, y_test)


# print('Coeficients:  ', *linear.coef_)
# print('Intercepts: ', linear.intercept_)

prediction = linear.predict(X_test)
for i in range(len(prediction)):
    print(f'Predicted value: {prediction[i]:.2f} Kg --->Actual value: {y_test[i]} Kg')
print(f'Accuracy -->  {accuracy}')