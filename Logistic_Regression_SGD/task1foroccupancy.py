import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

occupancy_detection = fetch_ucirepo(id=357)

X = occupancy_detection.data.features
y = occupancy_detection.data.targets
df = X.join(y)

df = df.drop('date', axis=1)
df = df.dropna(subset=['Occupancy'])

np.random.seed(31)
shuffled_df = df.sample(frac=1, random_state=31).reset_index(drop=True)

test_size = int(0.2 * len(df))
X_train = shuffled_df.drop('Occupancy', axis=1).iloc[:-test_size]
X_test = shuffled_df.drop('Occupancy', axis=1).iloc[-test_size:]
y_train = shuffled_df['Occupancy'].iloc[:-test_size]
y_test = shuffled_df['Occupancy'].iloc[-test_size:]

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
    X_test = np.array(X_test, dtype=np.float64)
    y_pred = sigmoid(np.dot(X_test, beta))
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_test * np.log(y_pred) + (1 - y_test) * np.log(1 - y_pred))

def sgd_with_loss_tracking_separate_plots(data, beta_init, X_test, y_test, max_epochs=1000, tolerance=1e-5, initial_lr=0.00001):
    beta = np.array(beta_init, dtype=np.float64)
    lr = initial_lr
    
    loss_differences = []
    test_log_losses = []
    iteration = 0
    
    for epoch in range(max_epochs):
        np.random.shuffle(data)
        
        for x, y in data:
            current_loss = log_likelihood_loss(beta, x, y)
            gradient = log_likelihood_grad(beta, x, y)
            beta_new = beta - lr * gradient
            new_loss = log_likelihood_loss(beta_new, x, y)
            
            loss_diff = abs(current_loss - new_loss)
            loss_differences.append(loss_diff)
            
            test_loss = log_loss_test(beta_new, X_test, y_test)
            test_log_losses.append(test_loss)
            
            if new_loss < current_loss:
                lr *= 1.05
                beta = beta_new
            else:
                lr *= 0.5
            
            iteration += 1
            
            if np.linalg.norm(lr * gradient) < tolerance:
                break
                
        if np.linalg.norm(lr * gradient) < tolerance:
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

beta_optimized, loss_differences, test_log_losses = sgd_with_loss_tracking_separate_plots(data_train, beta_init, X_test, y_test)

