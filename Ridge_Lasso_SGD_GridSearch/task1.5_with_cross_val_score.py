import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso, SGDRegressor
from sklearn.preprocessing import StandardScaler

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

best_ridge = grid_ridge.best_estimator_
best_lasso = grid_lasso.best_estimator_
best_sgd = grid_sgd.best_estimator_

ridge_scores = cross_val_score(best_ridge, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
lasso_scores = cross_val_score(best_lasso, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
sgd_scores = cross_val_score(best_sgd, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

ridge_rmse = np.sqrt(-ridge_scores)
lasso_rmse = np.sqrt(-lasso_scores)
sgd_rmse = np.sqrt(-sgd_scores)

model_scores = [ridge_rmse, lasso_rmse, sgd_rmse]
model_names = ['Ridge', 'Lasso', 'SGD']

plt.figure(figsize=(10, 6))
plt.boxplot(model_scores, vert=True, patch_artist=True, widths=0.6)
plt.xticks([1, 2, 3], model_names)
plt.ylabel('RMSE')
plt.title('Cross-Validation RMSE for Ridge, Lasso, and SGD Models')
plt.grid()
plt.tight_layout()
plt.show()

print(f"Ridge mean RMSE: {ridge_rmse.mean():.4f}")
print(f"Lasso mean RMSE: {lasso_rmse.mean():.4f}")
print(f"SGD mean RMSE: {sgd_rmse.mean():.4f}")
