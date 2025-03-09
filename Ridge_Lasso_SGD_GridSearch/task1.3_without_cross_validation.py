import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

D2 = pd.read_csv('winequality-red.csv', sep=';')

X = D2.drop(columns=['quality'])
y = D2['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "OLS": LinearRegression(),
    "Ridge (λ=0.01)": Ridge(alpha=0.01),
    "Ridge (λ=0.1)": Ridge(alpha=0.1),
    "Ridge (λ=1)": Ridge(alpha=1),
    "Lasso (λ=0.01)": Lasso(alpha=0.01, max_iter=10000),
    "Lasso (λ=0.1)": Lasso(alpha=0.1, max_iter=10000),
    "Lasso (λ=1)": Lasso(alpha=1, max_iter=10000),
    "SGD (λ=0.01)": SGDRegressor(alpha=0.01, max_iter=1000, tol=1e-3),
    "SGD (λ=0.1)": SGDRegressor(alpha=0.1, max_iter=1000, tol=1e-3),
    "SGD (λ=1)": SGDRegressor(alpha=1, max_iter=1000, tol=1e-3),
}

train_rmse = []
test_rmse = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))


plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(models))
plt.bar(index, train_rmse, bar_width, alpha=0.6, label='Train RMSE', color='blue')
plt.bar(index + bar_width, test_rmse, bar_width, alpha=0.6, label='Test RMSE', color='red')
plt.xticks(index + bar_width / 2, models.keys(), rotation=45, ha='right')
plt.ylabel('RMSE')
plt.title('RMSE Comparison of Different Models (Scaled Features)')
plt.legend()
plt.tight_layout()
plt.show()


