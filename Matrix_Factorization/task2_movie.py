import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def calculate_rmse(true_ratings, predicted_ratings):
    mask = true_ratings > 0
    return np.sqrt(mean_squared_error(true_ratings[mask], predicted_ratings[mask]))

def matrix_factorization_sgd(R, K, alpha, lambda_, epochs):
    n_users, n_items = R.shape
    P = np.random.normal(0, 0.1, (n_users, K))
    Q = np.random.normal(0, 0.1, (n_items, K))

    for epoch in range(epochs):
        for i in range(n_users):
            for j in range(n_items):
                if R[i, j] > 0:
                    error = R[i, j] - np.dot(P[i, :], Q[j, :])
                    P[i, :] += alpha * (error * Q[j, :] - lambda_ * P[i, :])
                    Q[j, :] += alpha * (error * P[i, :] - lambda_ * Q[j, :])

        P = np.clip(P, -1e3, 1e3)
        Q = np.clip(Q, -1e3, 1e3)

    return P, Q

def cross_validate(R, K, alpha, lambda_, epochs, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=31)
    rmse_scores = []

    for train_indices, test_indices in kf.split(R):
        R_train = np.zeros_like(R)
        R_test = np.zeros_like(R)

        for idx in train_indices:
            R_train[idx] = R[idx]
        for idx in test_indices:
            R_test[idx] = R[idx]

        P, Q = matrix_factorization_sgd(R_train, K, alpha, lambda_, epochs)
        R_pred = np.dot(P, Q.T)

        R_pred = np.nan_to_num(R_pred, nan=0.0, posinf=0.0, neginf=0.0)

        rmse = calculate_rmse(R_test, R_pred)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

def normalize_matrix(R):
    mean_user_ratings = np.divide(
        R.sum(axis=1), 
        (R != 0).sum(axis=1), 
        where=(R != 0).sum(axis=1) != 0
    )
    R_normalized = R.copy()
    for i in range(R.shape[0]):
        R_normalized[i, R[i, :] > 0] -= mean_user_ratings[i]
    return R_normalized, mean_user_ratings

def load_movielens_dataset(filepath):
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(filepath, sep='\t', names=column_names, engine='python')
    n_users = data['user_id'].nunique()
    n_items = data['item_id'].nunique()
    R = np.zeros((n_users, n_items))
    for row in data.itertuples():
        R[row.user_id - 1, row.item_id - 1] = row.rating
    return R

if __name__ == "__main__":
    data_file_path = './ml-100k/u.data'  
    R_movielens = load_movielens_dataset(data_file_path)
    R_movielens_normalized, _ = normalize_matrix(R_movielens)
    K = 10
    alpha = 0.005
    lambda_ = 0.1
    epochs = 20
    avg_rmse_movielens = cross_validate(R_movielens_normalized, K, alpha, lambda_, epochs)
    print(f"Average RMSE for MovieLens 100k Dataset: {avg_rmse_movielens:.4f}")
