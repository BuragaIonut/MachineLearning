import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('Titanic.csv', index_col = 'PassengerId')
print(df.head())

# # plt.hist('Fare', data = df)
# # plt.show()
# # x = np.array(df['Fare'])
# # sns.distplot(x)
# # # plt.show()
# # y = np.array(df['Age'])
# # sns.jointplot('Fare', 'Age', data = df)
# # plt.show()
# sns.violinplot(x='Embarked',y='Age', data = df)
# plt.show()