import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

X_iris = iris_df[iris.feature_names]
y_iris = iris_df['target']
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=31)

scaler_iris = StandardScaler()
X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)
X_test_iris_scaled = scaler_iris.transform(X_test_iris)

knn_param_grid_iris = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_grid_search_iris = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_param_grid_iris, cv=5, scoring='accuracy')
knn_grid_search_iris.fit(X_train_iris_scaled, y_train_iris)

knn_best_params_iris = knn_grid_search_iris.best_params_
knn_best_score_iris = knn_grid_search_iris.best_score_

tree_param_grid_iris = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 11),
    'min_samples_split': range(2, 11)
}
tree_grid_search_iris = GridSearchCV(estimator=DecisionTreeClassifier(random_state=31), param_grid=tree_param_grid_iris, cv=5, scoring='accuracy')
tree_grid_search_iris.fit(X_train_iris, y_train_iris)

tree_best_params_iris = tree_grid_search_iris.best_params_
tree_best_score_iris = tree_grid_search_iris.best_score_

knn_best_iris = KNeighborsClassifier(**knn_best_params_iris)
knn_best_iris.fit(X_train_iris_scaled, y_train_iris)
y_pred_knn_iris = knn_best_iris.predict(X_test_iris_scaled)
knn_test_accuracy_iris = accuracy_score(y_test_iris, y_pred_knn_iris)
print(classification_report(y_test_iris, y_pred_knn_iris))

tree_best_iris = DecisionTreeClassifier(**tree_best_params_iris, random_state=31)
tree_best_iris.fit(X_train_iris, y_train_iris)
y_pred_tree_iris = tree_best_iris.predict(X_test_iris)
tree_test_accuracy_iris = accuracy_score(y_test_iris, y_pred_tree_iris)
print(classification_report(y_test_iris, y_pred_tree_iris))

knn_cross_val_scores_iris = cross_val_score(knn_best_iris, X_train_iris_scaled, y_train_iris, cv=5, scoring='accuracy')
knn_cross_val_mean_iris = knn_cross_val_scores_iris.mean()

tree_cross_val_scores_iris = cross_val_score(tree_best_iris, X_train_iris, y_train_iris, cv=5, scoring='accuracy')
tree_cross_val_mean_iris = tree_cross_val_scores_iris.mean()

print(f"KNN - Best Hyperparameters: {knn_best_params_iris}, Cross-Validation Accuracy: {knn_cross_val_mean_iris:.2f}, Test Accuracy: {knn_test_accuracy_iris:.2f}")
print(f"Decision Tree - Best Hyperparameters: {tree_best_params_iris}, Cross-Validation Accuracy: {tree_cross_val_mean_iris:.2f}, Test Accuracy: {tree_test_accuracy_iris:.2f}")
