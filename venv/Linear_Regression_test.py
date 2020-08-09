import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

students = {'hours': [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43, 2, 39, 15, 44, 29, 41, 15, 24, 50],
            'test_results': [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73, 0, 62, 37, 74, 40, 90, 42, 58, 100]}

student_data = pd.DataFrame(data=students)
print(student_data)
x = student_data.hours
y = student_data.test_results

model = np.polyfit(x,y,1)
print(model)
plt.scatter(x,y)
plt.show()
