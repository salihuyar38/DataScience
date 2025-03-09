import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

D2 = pd.read_csv('winequality-red.csv', sep=';')

X = D2.drop(columns=['quality'])
y = D2['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
param_grid_lasso = {'alpha': [0.01, 0.1, 1, 10, 100]}
param_grid_sgd = {'alpha': [0.01, 0.1, 1, 10, 100], 'max_iter': [1000], 'tol': [1e-3]}

ridge = Ridge()
lasso = Lasso(max_iter=10000)
sgd = SGDRegressor()

grid_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, return_train_score=True, scoring='neg_mean_squared_error')
grid_lasso = GridSearchCV(lasso, param_grid_lasso, cv=5, return_train_score=True, scoring='neg_mean_squared_error')
grid_sgd = GridSearchCV(sgd, param_grid_sgd, cv=5, return_train_score=True, scoring='neg_mean_squared_error')

grid_ridge.fit(X_train_scaled, y_train)
grid_lasso.fit(X_train_scaled, y_train)
grid_sgd.fit(X_train_scaled, y_train)

def plot_cv_results(grid, title, param_name='alpha'):
    mean_train_score = -grid.cv_results_['mean_train_score']
    mean_test_score = -grid.cv_results_['mean_test_score']
    param_values = grid.cv_results_['param_' + param_name]
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, mean_train_score, label='Train MSE', marker='o')
    plt.plot(param_values, mean_test_score, label='Validation MSE', marker='o')
    plt.xscale('log')
    plt.xlabel(param_name)
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

plot_cv_results(grid_ridge, "Ridge Regression Cross-Validation Results")
plot_cv_results(grid_lasso, "Lasso Regression Cross-Validation Results")
plot_cv_results(grid_sgd, "SGD Regression Cross-Validation Results")
