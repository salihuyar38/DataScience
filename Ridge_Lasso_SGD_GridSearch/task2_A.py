import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = np.random.normal(1, 0.05, (100, 1))
rand = np.random.normal(1, 0.05, (100, 1))
y = 1.3 * (x ** 2) + 4.8 * x + 8 + rand

scaler = StandardScaler()
x_standardized = scaler.fit_transform(x)

degrees = [1, 2, 7, 10, 16, 100]

for degree in degrees:
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x_standardized)
    model = LinearRegression()
    model.fit(x_poly, y)
    
    y_pred = model.predict(x_poly)
    
    x_sorted = np.sort(x, axis=0).ravel()
    y_pred_sorted = y_pred[np.argsort(x, axis=0)].ravel()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='True Data', alpha=0.5)
    plt.plot(x_sorted, y_pred_sorted, color='red', label=f'Prediction (Degree {degree})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Prediction Curve for Polynomial Degree {degree}')
    plt.legend()
    plt.show()
