import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import Counter

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

X = df[iris.feature_names]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31)

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

def predict_knn(X_train, y_train, query_point, k=3, task="classification"):
    neighbors = top_k_nearest_neighbors(X_train, y_train, query_point, k)
    k_nearest_neighbors = [label for _, label in neighbors]
    if task == "classification":
        return Counter(k_nearest_neighbors).most_common(1)[0][0]
    elif task == "regression":
        return np.mean(k_nearest_neighbors)
    else:
        raise ValueError("Task must be either 'classification' or 'regression'")

k = 3
y_pred = [predict_knn(X_train_scaled, y_train, query, k=k, task="classification") for query in X_test_scaled]

accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Classification Accuracy with k={k}: {accuracy:.2f}")

print("\nSample Predictions:")
for i in range(5):
    print(f"True Label: {y_test.iloc[i]}, Predicted Label: {y_pred[i]}")
