import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

X_iris = iris_df[iris.feature_names]
y_iris = iris_df['target']
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=31)

scaler_iris = StandardScaler()
X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)
X_test_iris_scaled = scaler_iris.transform(X_test_iris)

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

def find_optimal_k_classification(X_train, y_train, X_test, y_test, max_k=20):
    errors = []
    for k in range(1, max_k + 1):
        y_pred = [predict_knn(X_train, y_train, query, k=k, task="classification") for query in X_test]
        error = 1 - accuracy_score(y_test, y_pred)
        errors.append(error)
    optimal_k = errors.index(min(errors)) + 1
    return errors, optimal_k

max_k = 20
iris_errors, iris_optimal_k = find_optimal_k_classification(X_train_iris_scaled, y_train_iris, X_test_iris_scaled, y_test_iris, max_k)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), iris_errors, marker='o', linestyle='--')
plt.title("Misclassification Rate vs. K (Iris Dataset)", fontsize=14)
plt.xlabel("Number of Neighbors (K)", fontsize=12)
plt.ylabel("Misclassification Rate", fontsize=12)
plt.xticks(range(1, max_k + 1))
plt.grid()
plt.show()

print(f"Optimal K for Iris dataset: {iris_optimal_k}")
