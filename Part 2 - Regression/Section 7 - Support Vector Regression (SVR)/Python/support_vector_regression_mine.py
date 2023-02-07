# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

y_scaled = y.reshape(len(y), 1)

print("Reshaped: \n")
print(y)

from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_scaled = scX.fit_transform(X)
scy = StandardScaler()
y_scaled = scy.fit_transform(y_scaled)

print("Scaled: \n")
print(X_scaled)
print(y_scaled)

# #Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

y_scaled = y_scaled.reshape(len(y))
print("Shaped again: \n")
print(y_scaled)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_scaled, y_scaled)

#predict
scaleInput = scX.transform([[6.5]])
scaledOutput = regressor.predict(scaleInput)
unscaledOutput = scy.inverse_transform(scaledOutput)

print(unscaledOutput)

# Visualising the Polynomial Regression results
plt.scatter(scX.inverse_transform(X_scaled), scy.inverse_transform(y_scaled), color = 'red')
plt.plot(scX.inverse_transform(X_scaled), scy.inverse_transform(regressor.predict(X_scaled)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(scX.inverse_transform(X_scaled)), max(scX.inverse_transform(X_scaled)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, scy.inverse_transform(y_scaled), color = 'red')
print("ZZOOT")
print(X_grid)
print(scX.inverse_transform(X_grid))
print(regressor.predict(scX.transform(X_grid)))
plt.plot(X_grid, scy.inverse_transform(regressor.predict(scX.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


