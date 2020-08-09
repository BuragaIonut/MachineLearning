import numpy as np
import pandas as pd
import sklearn

df = pd.read_csv('AirQualityUCI.csv', sep = ';')


df = df.drop(['Unnamed: 15','Unnamed: 16'], axis = 1)
print(df.isnull().sum())

print(df.head())