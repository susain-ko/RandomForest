import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#################################
### Linear Regression Attempt ###
#################################
###TO DO###

X = pd.read_csv("X.csv").values
Y = pd.read_csv("Y.csv", header=None).values
n = X.shape[0]


regr = linear_model.LinearRegression()
regr.fit(X, Y)

# Make predictions
Y_pred = regr.predict(X)

# Compute some statistics
res = Y - Y_pred
#standardized residuals
sig_hat = sqrt(sum(res**2)/(n-p-1))
h = X %*% solve(t(X) %*% X) %*% t(X)
h_diag = diag(h)
std_res = res/(sig_hat*sqrt(1-h_diag))
#predicted residuals
pred_res = res/(1-h_diag)
#RSS[i]
RSS = sum(res^2)
RSS_i = RSS - pred_res*res
#standardized predicted residuals
std_pred_res = (pred_res*sqrt(1 - h_diag))/sqrt(RSS_i/(n-p-2))
#Cook's distance
cook = std_res^2/(p+1)*h_diag/(1-h_diag)


# Make plots #
plt.figure()
plt.scatter(Y_pred, res)
plt.xlabel("fitted values")
plt.ylabel("residual")
plt.title("Residuals VS Fitted")
plt.show()

plt.plot(fitted, std_res, type="p")

# The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error: Training error
# print("Mean squared error: %.2f"
#       % mean_squared_error(Y, Y_pred))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(Y, Y_pred))