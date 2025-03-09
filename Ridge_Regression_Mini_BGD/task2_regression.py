import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import pandas as pd

wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets
df = X.join(y)

np.random.seed(31)
shuffled_df = df.sample(frac=1, random_state=31).reset_index(drop=True)
test_size = int(0.2 * len(df))
X_train = shuffled_df.drop('quality', axis=1).iloc[:-test_size].to_numpy()
X_test = shuffled_df.drop('quality', axis=1).iloc[-test_size:].to_numpy()
y_train = shuffled_df['quality'].iloc[:-test_size].to_numpy()
y_test = shuffled_df['quality'].iloc[-test_size:].to_numpy()

def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    return (X_train - mean) / std, (X_test - mean) / std

X_train, X_test = standardize(X_train, X_test)

def ridge_regression_mini_bgd(X, y, alpha, lmbda, batch_size=50, epochs=100):
    n_samples, n_features = X.shape
    w = np.random.randn(n_features)
    b = 0
    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled, y_shuffled = X[indices], y[indices]
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]
            predictions = X_batch.dot(w) + b
            errors = predictions - y_batch
            gradient_w = (2 / len(y_batch)) * X_batch.T.dot(errors) + 2 * lmbda * w
            gradient_b = (2 / len(y_batch)) * np.sum(errors)
            w -= alpha * gradient_w
            b -= alpha * gradient_b
    return w, b

def compute_rmse(X, y, w, b):
    predictions = X.dot(w) + b
    return np.sqrt(np.mean((y - predictions) ** 2))

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

alpha_values = [0.001, 0.01, 0.1]
lambda_values = [0.01, 0.1, 1]
batch_size = 50
epochs = 100
k = 5

results = []

folds = k_fold_split(X_train, y_train, k)
for alpha in alpha_values:
    for lmbda in lambda_values:
        rmse_scores = []
        for X_fold_train, y_fold_train, X_val, y_val in folds:
            X_fold_train, X_val = standardize(X_fold_train, X_val)
            w, b = ridge_regression_mini_bgd(X_fold_train, y_fold_train, alpha, lmbda, batch_size, epochs)
            rmse = compute_rmse(X_val, y_val, w, b)
            rmse_scores.append(rmse)
        mean_rmse = np.mean(rmse_scores)
        results.append((alpha, lmbda, mean_rmse))

results_df = pd.DataFrame(results, columns=["alpha", "lambda", "rmse"])
pivot_table = results_df.pivot(index="alpha", columns="lambda", values="rmse")

plt.figure(figsize=(10, 6))
plt.imshow(pivot_table, cmap="viridis", aspect="auto", origin="lower")
plt.colorbar(label="RMSE")
plt.xticks(range(len(lambda_values)), labels=lambda_values)
plt.yticks(range(len(alpha_values)), labels=alpha_values)
plt.xlabel("Lambda")
plt.ylabel("Alpha")
plt.title("Grid Search RMSE (5-Fold Cross-Validation)")
plt.show()

optimal_alpha = results_df.loc[results_df['rmse'].idxmin(), 'alpha']
optimal_lambda = results_df.loc[results_df['rmse'].idxmin(), 'lambda']

w, b = ridge_regression_mini_bgd(X_train, y_train, optimal_alpha, optimal_lambda, batch_size, epochs)

train_rmse_list = []
test_rmse_list = []
for epoch in range(epochs):
    train_rmse_list.append(compute_rmse(X_train, y_train, w, b))
    test_rmse_list.append(compute_rmse(X_test, y_test, w, b))

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_rmse_list, label="Train RMSE", linewidth=1.5)
plt.plot(range(1, epochs + 1), test_rmse_list, label="Test RMSE", linewidth=1.5)
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.title(f"RMSE for Training and Test Sets (Optimal α={optimal_alpha}, λ={optimal_lambda})")
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimal alpha: {optimal_alpha}, Optimal lambda: {optimal_lambda}")
