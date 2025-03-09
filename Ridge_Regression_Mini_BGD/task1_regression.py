import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets
df = X.join(y)

np.random.seed(31)
shuffled_df = df.sample(frac=1, random_state=31).reset_index(drop=True)
test_size = int(0.2 * len(df))
X_train = shuffled_df.drop('quality', axis=1).iloc[:-test_size]
X_test = shuffled_df.drop('quality', axis=1).iloc[-test_size:]
y_train = shuffled_df['quality'].iloc[:-test_size]
y_test = shuffled_df['quality'].iloc[-test_size:]

def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    return (X_train - mean) / std, (X_test - mean) / std

X_train_standardized, X_test_standardized = standardize(X_train, X_test)

def ridge_regression_mini_bgd(X, y, alpha, lmbda, batch_size=50, epochs=100):
    n_samples, n_features = X.shape
    w = np.random.randn(n_features)
    b = 0
    train_rmse_list = []
    test_rmse_list = []
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
        train_predictions = X.dot(w) + b
        train_rmse_list.append(np.sqrt(np.mean((y - train_predictions) ** 2)))
        test_predictions = X_test.dot(w) + b
        test_rmse_list.append(np.sqrt(np.mean((y_test - test_predictions) ** 2)))

    return w, b, train_rmse_list, test_rmse_list

alpha_values = [0.001, 0.01, 0.1]
lambda_values = [0.01, 0.1, 1]
batch_size = 50
epochs = 100

for alpha in alpha_values:
    for lmbda in lambda_values:
        w, b, train_rmse_list, test_rmse_list = ridge_regression_mini_bgd(
            X_train_standardized.to_numpy(), y_train.to_numpy(), alpha, lmbda, batch_size, epochs)
        plt.figure(figsize=(10, 6))
        plt.plot(range(epochs), train_rmse_list, label=f'Train RMSE (alpha={alpha}, lambda={lmbda})', color='blue')
        plt.plot(range(epochs), [-rmse for rmse in test_rmse_list], label=f'Test RMSE (alpha={alpha}, lambda={lmbda})', color='red')
        plt.title(f"RMSE for Training and Test Sets (alpha={alpha}, lambda={lmbda})")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()

