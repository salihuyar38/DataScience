import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def calculate_rmse(true_ratings, predicted_ratings):
    mask = true_ratings > 0
    return np.sqrt(mean_squared_error(true_ratings[mask], predicted_ratings[mask]))

def load_movielens_dataset(filepath):
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(filepath, sep='\t', names=column_names, engine='python')
    n_users = data['user_id'].nunique()
    n_items = data['item_id'].nunique()
    R = np.zeros((n_users, n_items))
    for row in data.itertuples():
        R[row.user_id - 1, row.item_id - 1] = row.rating
    return R

def perform_cross_validation(R, n_splits=3, n_components=10, max_iter=200, alpha_regularization=0.1, l1_ratio=0.5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []

    for train_indices, test_indices in kf.split(R):
        R_train = np.zeros_like(R)
        R_test = np.zeros_like(R)

        for idx in train_indices:
            R_train[idx] = R[idx]
        for idx in test_indices:
            R_test[idx] = R[idx]

        nmf = NMF(n_components=n_components, init='random', solver='cd', max_iter=max_iter, alpha_W=alpha_regularization, alpha_H=alpha_regularization, l1_ratio=l1_ratio, random_state=42)
        W = nmf.fit_transform(R_train)
        H = nmf.components_
        R_pred = np.dot(W, H)
        rmse = calculate_rmse(R_test, R_pred)
        rmse_scores.append(rmse)
    return np.mean(rmse_scores), rmse_scores

if __name__ == "__main__":
    data_file_path = './ml-100k/u.data'
    R_movielens = load_movielens_dataset(data_file_path)

    n_components = 10
    max_iter = 200
    alpha_regularization = 0.1
    l1_ratio = 0.5

    avg_rmse, rmse_scores = perform_cross_validation(R_movielens, n_splits=3, n_components=n_components, max_iter=max_iter, alpha_regularization=alpha_regularization, l1_ratio=l1_ratio)

    print(f"Average RMSE for MovieLens 100k Dataset using Coordinate Descent: {avg_rmse:.4f}")

    plt.figure()
    plt.bar(range(1, 4), rmse_scores)
    plt.title('RMSE for Each Fold')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.show()
