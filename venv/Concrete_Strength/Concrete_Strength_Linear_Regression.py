import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
data = pd.read_excel('Concrete_Data.xls', )

# print(data.head())

# print(data.isnull().sum())

# print(data.head())
data = data.drop(['fly_ash','blast_furnance_slag','superplasticizer'],1)
predict = 'concrete_strength'

X = np.array(data.drop(['concrete_strength'], axis = 1))
y = np.array(data[predict])
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.33,random_state=50)

model = LinearRegression()
model.fit(X_train,y_train)

# print(model.score(X_test,y_test))

prediction = model.predict(X_test)

for i in range(len(prediction)):
    print(f'Actual strength: {y_test[i]:.2f} ---> Predicted value: {prediction[i]:.2f}')
print(f'Accuracy: {model.score(X_test,y_test)*100:.2f} %')