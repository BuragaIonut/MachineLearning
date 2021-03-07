import pandas as pd
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn.metrics import roc_curve

df = pd.read_csv('FlightData.csv')
# print(df.head())
# print(df.isnull().sum())

df = df.drop('Unnamed: 25',axis = 1)
# print(df.isnull().sum())

df = df[["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]]
# print(df.isnull().sum())

# print(df[df.isnull().values.any(axis=1)].head())

df = df.fillna({'ARR_DEL15': 1})
# print(df.iloc[177:185])

import math

for index, row in df.iterrows():
    df.loc[index, 'CRS_DEP_TIME'] = math.floor(row['CRS_DEP_TIME'] / 100)
# print(df.head())

df = pd.get_dummies(df, columns=['ORIGIN', 'DEST'])
# print(df.head())


train_x, test_x, train_y, test_y = train_test_split(df.drop('ARR_DEL15', axis=1), df['ARR_DEL15'], test_size=0.2, random_state=42)
# print(train_x.shape)
# print(test_x.shape)

model = RandomForestClassifier(random_state=13)
model.fit(train_x, train_y)

predicted = model.predict(test_x)
# print(model.score(test_x,test_y))

probabilities = model.predict_proba(test_x)
# print(roc_auc_score(test_y, probabilities[:, 1]))

# print(confusion_matrix(test_y, predicted))

train_predictions = model.predict(train_x)
# print(precision_score(train_y, train_predictions))

# print(recall_score(train_y, train_predictions))

sns.set()



fpr, tpr, _ = roc_curve(test_y, probabilities[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


def predict_delay(departure_date_time, origin, destination):
    from datetime import datetime

    try:
        departure_date_time_parsed = datetime.strptime(departure_date_time, '%d/%m/%Y %H:%M:%S')
    except ValueError as e:
        return 'Error parsing date/time - {}'.format(e)

    month = departure_date_time_parsed.month
    day = departure_date_time_parsed.day
    day_of_week = departure_date_time_parsed.isoweekday()
    hour = departure_date_time_parsed.hour

    origin = origin.upper()
    destination = destination.upper()

    input = [{'MONTH': month,
              'DAY': day,
              'DAY_OF_WEEK': day_of_week,
              'CRS_DEP_TIME': hour,
              'ORIGIN_ATL': 1 if origin == 'ATL' else 0,
              'ORIGIN_DTW': 1 if origin == 'DTW' else 0,
              'ORIGIN_JFK': 1 if origin == 'JFK' else 0,
              'ORIGIN_MSP': 1 if origin == 'MSP' else 0,
              'ORIGIN_SEA': 1 if origin == 'SEA' else 0,
              'DEST_ATL': 1 if destination == 'ATL' else 0,
              'DEST_DTW': 1 if destination == 'DTW' else 0,
              'DEST_JFK': 1 if destination == 'JFK' else 0,
              'DEST_MSP': 1 if destination == 'MSP' else 0,
              'DEST_SEA': 1 if destination == 'SEA' else 0}]

    return model.predict_proba(pd.DataFrame(input))[0][0]
print(f"Chances to arrive on time: {predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL')*100}%")
print(f"Chances to arrive on time: {predict_delay('1/10/2017 12:45:00', 'ATL', 'JFK')*100}%")
