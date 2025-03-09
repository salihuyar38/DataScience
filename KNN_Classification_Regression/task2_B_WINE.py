import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

wine_df = pd.read_csv("winequality-red.csv", delimiter=';')

if wine_df.isnull().sum().any():
    wine_df.fillna(wine_df.mean(), inplace=True)

if np.isinf(wine_df.values).sum() > 0:
    wine_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    wine_df.fillna(wine_df.mean(), inplace=True)

X_wine = wine_df.drop(columns=['quality'])
y_wine = wine_df['quality']
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=31, stratify=y_wine
)

scaler_wine = StandardScaler()
X_train_wine_scaled = scaler_wine.fit_transform(X_train_wine)
X_test_wine_scaled = scaler_wine.transform(X_test_wine)

knn_param_grid = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_param_grid, cv=5, scoring='accuracy')
knn_grid_search.fit(X_train_wine_scaled, y_train_wine)

knn_best_params = knn_grid_search.best_params_
knn_best_score = knn_grid_search.best_score_

tree_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 11),
    'min_samples_split': range(2, 11)
}
tree_grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=31), param_grid=tree_param_grid, cv=5, scoring='accuracy')
tree_grid_search.fit(X_train_wine, y_train_wine)

tree_best_params = tree_grid_search.best_params_
tree_best_score = tree_grid_search.best_score_

knn_best = KNeighborsClassifier(**knn_best_params)
knn_best.fit(X_train_wine_scaled, y_train_wine)
y_pred_knn = knn_best.predict(X_test_wine_scaled)
print(classification_report(y_test_wine, y_pred_knn, zero_division=0))

tree_best = DecisionTreeClassifier(**tree_best_params, random_state=31)
tree_best.fit(X_train_wine, y_train_wine)
y_pred_tree = tree_best.predict(X_test_wine)
print(classification_report(y_test_wine, y_pred_tree, zero_division=0))

knn_cross_val_scores = cross_val_score(knn_best, X_train_wine_scaled, y_train_wine, cv=5, scoring='accuracy')
knn_cross_val_mean = knn_cross_val_scores.mean()

tree_cross_val_scores = cross_val_score(tree_best, X_train_wine, y_train_wine, cv=5, scoring='accuracy')
tree_cross_val_mean = tree_cross_val_scores.mean()

print(f"KNN - Best Hyperparameters: {knn_best_params}, Cross-Validation Accuracy: {knn_cross_val_mean:.2f}, Test Accuracy: {accuracy_score(y_test_wine, y_pred_knn):.2f}")
print(f"Decision Tree - Best Hyperparameters: {tree_best_params}, Cross-Validation Accuracy: {tree_cross_val_mean:.2f}, Test Accuracy: {accuracy_score(y_test_wine, y_pred_tree):.2f}")
