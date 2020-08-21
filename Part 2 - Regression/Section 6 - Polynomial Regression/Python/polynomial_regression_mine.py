import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_polly = poly_reg.fit_transform(X)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_polly, y)

lin_reg_pred = lin_reg.predict([[6.5]])
print(lin_reg_pred)
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Truth or Bluff(Linear Reg)")
plt.xlabel("Potition Level")
plt.ylabel("Salary")
plt.show()

# lin_reg_poly_pred = lin_reg_poly.predict([[6.5, 6.5*6.5, 6.5*6.5*6.5, 6.5*6.5*6.5*6.5, 6.5*6.5*6.5*6.5*6.5]])
# print(lin_reg_poly_pred)
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_poly.predict(X_polly), color='blue')
plt.title("Truth or Bluff(Poly Reg)")
plt.xlabel("Potition Level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_poly.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg_poly_pred = lin_reg_poly.predict(poly_reg.fit_transform([[6.5]]))
print(lin_reg_poly_pred)

print(lin_reg_poly.coef_)