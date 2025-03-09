import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

file_path = "winequality-red.csv"
wine_df = pd.read_csv(file_path, delimiter=';')
if wine_df.isnull().sum().any():
    wine_df.fillna(wine_df.mean(), inplace=True)

if np.isinf(wine_df.values).sum() > 0:
    wine_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    wine_df.fillna(wine_df.mean(), inplace=True)


X_wine = wine_df.drop(columns=['quality'])
y_wine = wine_df['quality']

X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.3, random_state=31)

scaler_wine = StandardScaler()
X_train_wine_scaled = scaler_wine.fit_transform(X_train)
X_test_wine_scaled = scaler_wine.transform(X_test)

def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))

def top_k_nearest_neighbors(X_train, y_train, query_point, k=3):
    distances = []
    for train_row, train_label in zip(X_train, y_train):
        distance = euclidean_distance(query_point, train_row)
        distances.append((distance, train_label))
    distances.sort(key=lambda x: x[0])
    return distances[:k]

def predict_knn(X_train, y_train, query_point, k=3, task="regression"):
    neighbors = top_k_nearest_neighbors(X_train, y_train, query_point, k)
    k_nearest_neighbors = [label for _, label in neighbors]
    if task == "regression":
        return np.mean(k_nearest_neighbors)

def find_optimal_k_regression(X_train, y_train, X_test, y_test, max_k=20):
    errors = []
    for k in range(1, max_k + 1):
        y_pred = [predict_knn(X_train, y_train, query, k=k, task="regression") for query in X_test]
        error = mean_squared_error(y_test, y_pred)
        errors.append(error)
    optimal_k = errors.index(min(errors)) + 1
    return errors, optimal_k

max_k = 20
wine_errors, wine_optimal_k = find_optimal_k_regression(X_train_wine_scaled, y_train, X_test_wine_scaled, y_test, max_k)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wine_errors, marker='o', linestyle='--')
plt.title("Mean Squared Error vs. K (Wine Quality Dataset)", fontsize=14)
plt.xlabel("Number of Neighbors (K)", fontsize=12)
plt.ylabel("Mean Squared Error", fontsize=12)
plt.xticks(range(1, max_k + 1))
plt.grid()
plt.show()

print(f"Optimal K for Wine Quality dataset: {wine_optimal_k}")
