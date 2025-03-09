import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('bank.csv', sep=';')
data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
data['y'] = data['y'].map({'yes': 1, 'no': 0})
data = data.astype(int)

X = data.drop('y', axis=1).values
y = data['y'].values

def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_logistic(X, y, theta, lambda_):
    m = len(y)
    predictions = sigmoid(X @ theta)
    gradient = (1/m) * X.T @ (predictions - y)
    gradient[1:] += (lambda_ / m) * theta[1:]
    return gradient

def logistic_ridge_mini_bgd(X_train, y_train, alpha, lambda_, batch_size, num_epochs):
    m, n = X_train.shape
    theta = np.zeros(n)
    rmse_train = []
    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        for i in range(0, m, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            gradient = gradient_logistic(X_batch, y_batch, theta, lambda_)
            theta -= alpha * gradient
        train_predictions = sigmoid(X_train @ theta)
        rmse_train.append(np.sqrt(np.mean((train_predictions - y_train) ** 2)))
    return theta, rmse_train

def compute_rmse(X, y, theta):
    predictions = sigmoid(X @ theta)
    residuals = predictions - y
    return np.sqrt(np.mean(residuals ** 2))

def k_fold_split(X, y, k):
    fold_size = len(X) // k
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
        y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)
        folds.append((X_train, y_train, X_val, y_val))
    return folds

alphas = [0.001, 0.01, 0.1]
lambdas = [0.01, 0.1, 1.0]
batch_size = 50
num_epochs = 50
k = 5

results = []

folds = k_fold_split(X, y, k)
for alpha in alphas:
    for lambda_ in lambdas:
        rmse_scores = []
        for X_train, y_train, X_val, y_val in folds:
            X_train, X_val = standardize(X_train, X_val)
            theta = logistic_ridge_mini_bgd(X_train, y_train, alpha, lambda_, batch_size, num_epochs)[0]
            rmse = compute_rmse(X_val, y_val, theta)
            rmse_scores.append(rmse)
        mean_rmse = np.mean(rmse_scores)
        results.append((alpha, lambda_, mean_rmse))

results_df = pd.DataFrame(results, columns=["alpha", "lambda", "rmse"])
pivot_table = results_df.pivot(index="alpha", columns="lambda", values="rmse")

plt.figure(figsize=(10, 6))
plt.imshow(pivot_table, cmap="viridis", aspect="auto", origin="lower")
plt.colorbar(label="RMSE")
plt.xticks(range(len(lambdas)), labels=lambdas)
plt.yticks(range(len(alphas)), labels=alphas)
plt.xlabel("Lambda")
plt.ylabel("Alpha")
plt.title("Grid Search RMSE (5-Fold Cross-Validation)")
plt.show()

optimal_alpha = results_df.loc[results_df['rmse'].idxmin(), 'alpha']
optimal_lambda = results_df.loc[results_df['rmse'].idxmin(), 'lambda']

np.random.seed(31)
shuffled_indices = np.random.permutation(len(X))
test_size = int(0.2 * len(X))
train_indices = shuffled_indices[:-test_size]
test_indices = shuffled_indices[-test_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

X_train, X_test = standardize(X_train, X_test)

theta, rmse_train = logistic_ridge_mini_bgd(X_train, y_train, optimal_alpha, optimal_lambda, batch_size, num_epochs)

rmse_test = []
for epoch in range(num_epochs):
    predictions_test = sigmoid(X_test @ theta)
    rmse_test.append(np.sqrt(np.mean((predictions_test - y_test) ** 2)))

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), rmse_train, label="Train RMSE", linewidth=1.5)
plt.plot(range(1, num_epochs + 1), rmse_test, label="Test RMSE", linewidth=1.5)
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.title(f"RMSE for Training and Test Sets (Optimal α={optimal_alpha}, λ={optimal_lambda})")
plt.legend()
plt.grid(True)
plt.show()
print(f"Optimal alpha: {optimal_alpha}, Optimal lambda: {optimal_lambda}")
