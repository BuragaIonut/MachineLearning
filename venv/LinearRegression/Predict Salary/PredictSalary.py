import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prediction(years):
    salary_prediction = linear.predict([[years]])
    return salary_prediction
df = pd.read_csv('dataset.csv')


# print(df.head())

X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

linear = LinearRegression()

linear.fit(X_train, y_train)

predicted_salary = linear.predict(X_test)

acc = linear.score(X_test,y_test)
# print(acc)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, linear.predict(X_train), color='blue')


years = float(input('How many years of experience do u have?'))
print(f'{float(prediction(years)):.2f} is the predicted salary for {years} years of experience')

