import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = pd.read_csv('Advertising.csv')

###############SIMPLE LINEAR REGRESSION########################


# print(data.head())


# data = data.drop('Unnamed: 0', axis = 1)
#
# print(data.head())
#
#
# plt.figure(figsize=(16, 8))
# plt.scatter(
#     data['TV'],
#     data['sales'],
#     c='black'
# )
# plt.xlabel("Money spent on TV ads ($)")
# plt.ylabel("Sales ($)")
# # plt.show()
#
# x = data['TV'].values.reshape(-1,1)
# y = data['sales'].values.reshape(-1,1)
#
# linear = LinearRegression()
# linear.fit(x,y)
#
# prediction = linear.predict(x)
#
# plt.plot(
#     data['TV'],
#     prediction,
#     c='blue',
#     linewidth=2
# )
# # plt.show()


##############MULTIPLE LINEAR REGRESSION###################


xs = data.drop(['sales','Unnamed: 0'], axis = 1)
y = data['sales'].values.reshape(-1,1)

linear = LinearRegression()
linear.fit(xs,y)

prediction = linear.predict(xs)
print(linear.score(xs,y)
