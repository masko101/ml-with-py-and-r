# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)
print(y_predict)

plt.scatter(X_train, y_train, color='red')
y_train_predict = lr.predict(X_train)
plt.plot(X_train, y_train_predict, color='blue')
plt.title("Salary vs Experience Training")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, y_train_predict, color='blue')
plt.title("Salary vs Experience Test")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()

myQuery = [35]
myPred = lr.predict([myQuery])
intercept_ = lr.intercept_
fPred = lr.coef_ * myQuery + intercept_
print(fPred)