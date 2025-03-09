import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import Counter

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
    if task == "classification":
        return Counter(k_nearest_neighbors).most_common(1)[0][0]
    elif task == "regression":
        return np.mean(k_nearest_neighbors)
    else:
        raise ValueError("Task must be either 'classification' or 'regression'")

k = 3
y_pred = [predict_knn(X_train_scaled, y_train, query, k=k, task="regression") for query in X_test_scaled]

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"KNN Regression Evaluation with k={k}:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

print("\nSample Predictions:")
for i in range(5):
    print(f"True Quality: {y_test.iloc[i]}, Predicted Quality: {y_pred[i]:.2f}")
