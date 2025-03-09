import pandas as pd  
from sklearn.model_selection import train_test_split
import numpy as np 
import matplotlib.pyplot as plt 
import scipy
from scipy.linalg import cholesky,solve,solve_triangular

Xdata =pd.read_csv('GasPrices.csv')



#Drop the other columns which not realy helping to our prediction
xdata = Xdata[['Price', 'Pumps', 'Interior', 'Restaurant', 'CarWash', 'Highway', 'Intersection', 'Gasolines']]
ydata = Xdata['Price']

xdata = pd.get_dummies(xdata)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(xdata, ydata, test_size=0.2, random_state=31)
def gaussian_elimination(A, b):
    n = len(b)
    for i in range(n):
        max_index = np.argmax(np.abs(A[i:, i])) + i
        if A[max_index, i] == 0:
            raise ValueError("Error")
       
        if max_index != i:
            A[[i, max_index]] = A[[max_index, i]]
            b[i], b[max_index] = b[max_index], b[i]

        A[i] = A[i] / A[i, i]
        b[i] = b[i] / A[i, i]
        for j in range(i + 1, n):
            factor = A[j, i]
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]

    # Do back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.sum(A[i, i + 1:] * x[i + 1:])
    return x 

def cholesky_decomposition(A, b):
    L = cholesky(A, lower=True)
    y = solve_triangular(L, b, lower=True)
    x = solve_triangular(L.T, y, lower=False)
    return x

def qr_decomposition(A, b):
    Q, R = np.linalg.qr(A)
    # Solve Rx = Q^Tb
    x = np.linalg.solve(R, Q.T @ b)
    return x
# QR decomposition implementation
def learn_linreg_NormEq(X, y, method='gaussian',regularization=1e-10):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  
    A = np.asarray(X_b.T @ X_b, dtype=np.float64) + np.eye(X_b.shape[1]) * regularization
    b = np.asarray(X_b.T @ y, dtype=np.float64)
    
    
   # A = X_b.T @ X_b
    #b = X_b.T @ y
    if method == 'gaussian':
        beta_hat = gaussian_elimination(A, b)
    elif method == 'cholesky':
        # Perform Cholesky decomposition
        L = cholesky(A, lower=True)
        # Solve Ly = b for y
        y = solve_triangular(L, b, lower=True)
        # Solve L^T x = y for x
        beta_hat = solve_triangular(L.T, y, lower=False)
    elif method == 'decomposition':
        beta_hat = qr_decomposition(A, b)
    else:
        raise ValueError("Error")
    
    return beta_hat

    
beta_gaussian = learn_linreg_NormEq(Xtrain, Ytrain, method='gaussian')
beta_cholesky=learn_linreg_NormEq(Xtrain, Ytrain, method='cholesky')
beta_decomposition=learn_linreg_NormEq(Xtrain, Ytrain, method='decomposition')


def predict(X, beta):
   
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  
    return X_b @ beta

# Predictions
prediction_gaussian = predict(Xtest, beta_gaussian)
prediction_cholesky = predict(Xtest, beta_cholesky)
prediction_decomposition = predict(Xtest, beta_decomposition)
#print(prediction_decomposition)

#print(prediction_cholesky)



def evaluate_model(y_true, y_pred):
    residuals = np.abs(y_true - y_pred)
    return residuals



residuals_gaussian= evaluate_model(Ytest, prediction_gaussian)
residuals_cholesky= evaluate_model(Ytest, prediction_cholesky)
residuals_decomposition= evaluate_model(Ytest, prediction_decomposition)

def average_residual(y_true, y_pred):
    
    residuals = np.abs(y_true - y_pred)
    avg_residual = np.mean(residuals)
    return avg_residual

avg_residual_gaussian = average_residual(Ytest, prediction_gaussian)
avg_residual_cholesky = average_residual(Ytest, prediction_cholesky)
avg_residual_decomposition = average_residual(Ytest, prediction_decomposition)


def rmse(y_true, y_pred):
    squared_diffs = (y_true - y_pred) ** 2
    rmse_value = np.sqrt(np.mean(squared_diffs))
    return rmse_value

rmse_gaussian = rmse(Ytest, prediction_gaussian)
rmse_cholesky = rmse(Ytest, prediction_cholesky)
rmse_qr = rmse(Ytest, prediction_decomposition)


print("RMSE (Gaussian Elimination):", rmse_gaussian)
print("RMSE (Cholesky Decomposition):", rmse_cholesky)
print("RMSE (QR Decomposition):", rmse_qr)

'''
plt.scatter(Ytest, residuals_gaussian)
plt.title('Residuals (Gaussian)')
plt.xlabel('True Values')
plt.ylabel('Residuals')

# Plot for Cholesky method

plt.scatter(Ytest, residuals_cholesky)
plt.title('Residuals (Cholesky)')
plt.xlabel('True Values')
plt.ylabel('Residuals')



# Plot for QR method

plt.scatter(Ytest, residuals_decomposition)
plt.title('Residuals (QR)')
plt.xlabel('True Values')
plt.ylabel('Residuals')


plt.tight_layout()
plt.show()
'''


