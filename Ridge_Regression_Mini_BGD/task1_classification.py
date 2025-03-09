import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('bank.csv', sep=';')



data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
data['y'] = data['y'].map({'yes': 1, 'no': 0})
data = data.astype(int)

np.random.seed(31)
shuffled_df = data.sample(frac=1, random_state=31).reset_index(drop=True)
test_size = int(0.2 * len(data))

X = shuffled_df.drop('y', axis=1).values
y = shuffled_df['y'].values
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test

X_train, X_test = standardize(X_train, X_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost_function_logistic(X, y, theta, lambda_):
    m = len(y)
    predictions = sigmoid(X @ theta)
    cost = -(1/m) * (y @ np.log(predictions) + (1 - y) @ np.log(1 - predictions))
    reg_term = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    return cost + reg_term
def gradient_logistic(X, y, theta, lambda_):
    m = len(y)
    predictions = sigmoid(X @ theta)
    gradient = (1/m) * X.T @ (predictions - y)
    gradient[1:] += (lambda_ / m) * theta[1:]
    return gradient
def logistic_ridge_mini_bgd(X_train, y_train, X_test, y_test, alpha, lambda_, batch_size, num_epochs):
    m, n = X_train.shape
    theta = np.zeros(n)
    rmse_train = []
    rmse_test = []
    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        for i in range(0, m, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            gradient = gradient_logistic(X_batch, y_batch, theta, lambda_)
            theta -= alpha * gradient
        rmse_train.append(compute_rmse(X_train, y_train, theta))
        rmse_test.append(compute_rmse(X_test, y_test, theta))

    return theta, rmse_train, rmse_test

def compute_rmse(X, y, theta):
    predictions = sigmoid(X @ theta)
    residuals = predictions - y
    return np.sqrt(np.mean(residuals ** 2))

def plot_rmse_enhanced(rmse_train, rmse_test, alpha, lambda_, num_epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), rmse_train, label="Train RMSE", linewidth=1)
    plt.plot(range(1, num_epochs + 1), [-x for x in rmse_test], label="Test RMSE (Negative)", linewidth=1)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.ylim(-0.5, 0.5)  
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f"RMSE Convergence (alpha={alpha}, lambda={lambda_})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

alphas = [0.001, 0.01, 0.1]
lambdas = [0.01, 0.1, 1.0]
batch_size = 50
num_epochs = 100

for alpha in alphas:
    for lambda_ in lambdas:
        theta, rmse_train, rmse_test = logistic_ridge_mini_bgd(X_train, y_train, X_test, y_test, alpha, lambda_, batch_size, num_epochs)
        plot_rmse_enhanced(rmse_train, rmse_test, alpha, lambda_, num_epochs)
