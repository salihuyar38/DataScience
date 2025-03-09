import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features
y = bank_marketing.data.targets

y = y['y']
y = y.map({'yes': 1, 'no': 0})

df = X.join(y)
df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
df = df.astype(int)

np.random.seed(31)
shuffled_df = df.sample(frac=1, random_state=31).reset_index(drop=True)

test_size = int(0.2 * len(df))

X_train = shuffled_df.drop('y', axis=1).iloc[:-test_size]
X_test = shuffled_df.drop('y', axis=1).iloc[-test_size:]
y_train = shuffled_df['y'].iloc[:-test_size]
y_test = shuffled_df['y'].iloc[-test_size:]

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

data_train = [(np.array(x), np.array(y)) for x, y in zip(X_train, y_train)]

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def log_likelihood_loss(beta, x, y, epsilon=1e-15):
    beta = np.array(beta, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y_pred = sigmoid(np.dot(x, beta))
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

def log_likelihood_grad(beta, x, y):
    beta = np.array(beta, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y_pred = sigmoid(np.dot(x, beta))
    return (y_pred - y) * x

def log_loss_test(beta, X_test, y_test, epsilon=1e-15):
    y_pred = sigmoid(np.dot(X_test, beta))
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_test * np.log(y_pred) + (1 - y_test) * np.log(1 - y_pred))

def sgd_with_adagrad(data, beta_init, X_test, y_test, max_epochs=1000, tolerance=1e-5, initial_lr=0.001):
    beta = np.array(beta_init, dtype=np.float64)
    h = np.zeros_like(beta)
    epsilon = 1e-8
    loss_differences = []
    test_log_losses = []
    iteration = 0
    for epoch in range(max_epochs):
        np.random.shuffle(data)
        for x, y in data:
            current_loss = log_likelihood_loss(beta, x, y)
            gradient = log_likelihood_grad(beta, x, y)
           
            h += gradient ** 2
            adjusted_lr = initial_lr / (np.sqrt(h) + epsilon)
            beta -= adjusted_lr * gradient
            
            new_loss = log_likelihood_loss(beta, x, y)
            
            loss_diff = abs(current_loss - new_loss)
            loss_differences.append(loss_diff)
            
            test_loss = log_loss_test(beta, X_test, y_test)
            test_log_losses.append(test_loss)
            
            iteration += 1
            
            if np.linalg.norm(adjusted_lr * gradient) < tolerance:
                break
                
        if np.linalg.norm(adjusted_lr * gradient) < tolerance:
            print(f"Converged at epoch {epoch}")
            break

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(loss_differences)), loss_differences, color='tab:blue')
    plt.xlabel('Iteration')
    plt.ylabel('|f(xᵢ₋₁) - f(xᵢ)|')
    plt.title('Loss Difference vs Iteration')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(test_log_losses)), test_log_losses, color='tab:orange')
    plt.xlabel('Iteration')
    plt.ylabel('Test Log-Loss')
    plt.title('Test Log-Loss vs Iteration')
    plt.grid(True)
    plt.show()

    return beta, loss_differences, test_log_losses

beta_init = np.zeros(X_train.shape[1])
data_train = [(np.array(x), np.array(y)) for x, y in zip(X_train, y_train)]

beta_optimized, loss_differences, test_log_losses = sgd_with_adagrad(data_train, beta_init, X_test, y_test)
