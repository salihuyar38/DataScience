import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)
X_wine = wine_quality.data.features
y_wine = wine_quality.data.targets
data_combined_wein = pd.concat([X_wine, y_wine], axis=1).sample(500, random_state=31)

print(data_combined_wein.head())


def split_set(data):
    data = data.sample(frac=1, random_state=31).reset_index(drop=True)
    index = int(len(data) * 0.8)
    train_data = data.iloc[:index]
    test_data = data.iloc[index:]
    return train_data, test_data

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def steplength_armijo(f, x, d, delta=0.5):
    alpha = 1
    while f(x) - f(x + alpha * d) < alpha * delta * d.T @ d:
        alpha /= 2
    return alpha

def steplength_bolddriver(f, x, d, alpha_old=1, alpha_plus=1.1, alpha_minus=0.5):
    alpha = alpha_old * alpha_plus
    while f(x) - f(x + alpha * d) <= 0:
        alpha *= alpha_minus
    return alpha

def minimize_gd(f, x0, X, y, alpha, imax, epsilon, step_method='armijo'):
    x = x0
    history = []
    alpha_old = alpha

    for i in range(imax):
        gradient = -2 * X.T @ (y - X @ x)
        
        if step_method == 'armijo':
            step_length = steplength_armijo(f, x, gradient)
        elif step_method == 'bolddriver':
            step_length = steplength_bolddriver(f, x, gradient, alpha_old)
            alpha_old = step_length
        
        x_new = x + step_length * gradient
        history.append(f(x_new))
        
        if abs(f(x) - f(x_new)) < epsilon:
            return x_new, history
        x = x_new

    raise ValueError("Did not converge in the maximum number of iterations")

def learn_linreg_gd(X_train, y_train, alpha, imax, epsilon, step_method='armijo'):
    X = np.c_[np.ones(X_train.shape[0]), X_train]
    beta_hat = np.zeros(X.shape[1])
    
    def loss(beta_hat):
        return np.sum((y_train - X @ beta_hat) ** 2)
    
    beta_hat, history = minimize_gd(loss, beta_hat, X, y_train, alpha, imax, epsilon, step_method)
    return beta_hat, history

def learn_linreg_gd_fixed_step(X_train, y_train, alpha, imax, epsilon):
    X = np.c_[np.ones(X_train.shape[0]), X_train]
    beta_hat = np.zeros(X.shape[1])
    
    def loss(beta_hat):
        return np.sum((y_train - X @ beta_hat) ** 2)
    
    history = []
    for i in range(imax):
        gradient = -2 * X.T @ (y_train - X @ beta_hat)
        beta_hat_new = beta_hat + alpha * gradient
        history.append(loss(beta_hat_new))
        
        if abs(loss(beta_hat) - loss(beta_hat_new)) < epsilon:
            break
        beta_hat = beta_hat_new
        
    return beta_hat, history

wine_train, wine_test = split_set(data_combined_wein)

X_train_wine = wine_train.iloc[:, :-1].values
y_train_wine = wine_train.iloc[:, -1].values
X_test_wine = wine_test.iloc[:, :-1].values
y_test_wine = wine_test.iloc[:, -1].values
X_test_wine_bias = np.c_[np.ones(X_test_wine.shape[0]), X_test_wine]

alpha = 0.01
imax = 50
epsilon = 1e-5  

fixed_alphas = [0.01, 0.05, 0.1]
fixed_histories = {}
for fixed_alpha in fixed_alphas:
    _, history = learn_linreg_gd_fixed_step(X_train_wine, y_train_wine, fixed_alpha, imax, epsilon)
    fixed_histories[f"Fixed Step Length {fixed_alpha}"] = history

beta_hat_armijo, history_armijo = learn_linreg_gd(X_train_wine, y_train_wine, alpha, imax, epsilon, 'armijo')
beta_hat_bolddriver, history_bolddriver = learn_linreg_gd(X_train_wine, y_train_wine, alpha, imax, epsilon, 'bolddriver')

plt.figure(figsize=(12, 8))
for label, history in fixed_histories.items():
    plt.plot(range(len(history)), history, label=label)
plt.plot(range(len(history_armijo)), history_armijo, label="StepLength-Armijo", linestyle="--")
plt.plot(range(len(history_bolddriver)), history_bolddriver, label="StepLength-BoldDriver", linestyle="--")

plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("RMSE Comparison for Different Step Length Algorithms")
plt.legend()
plt.grid()
plt.show()
