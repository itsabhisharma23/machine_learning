import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

class LassoRegularizer:
    """Class to implement Lasso (L1) Regularization using Proximal Gradient Descent"""

    def __init__(self):
        self._lambda = None
        self._alpha = None
        self._W = None
        self._b = None

    def _soft_threshold(self, w, thresh):
        """Soft-thresholding operator for L1 regularization"""
        return np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)

    def fit(self, X, y, alpha, lam, iterations=1000):
        """Fit the data using Proximal Gradient Descent"""
        m, n_features = X.shape
        self._lambda = lam
        self._alpha = alpha

        print("Gradient Descent Implementation of Lasso Regression")
        
        # Initialize weights and bias
        self._W = np.zeros(n_features)
        self._b = 0.0

        for i in range(iterations):
            y_h = self.predict(X)
            
            # Cost calculation (MSE + L1 penalty)
            cost = np.sum((y_h - y)**2) / (2 * m) + (self._lambda / m) * np.sum(np.abs(self._W))
            
            if i % (iterations // 5 if iterations >= 5 else 1) == 0:
                print(f"Cost after {i} iterations is {cost:06.4f}")
            
            # Standard gradient for MSE part
            dW = np.dot(X.T, (y_h - y)) / m
            db = np.sum(y_h - y) / m

            # Gradient step for weights (without L1 term yet)
            w_temp = self._W - self._alpha * dW

            # Apply Proximal Operator (Soft-Thresholding) for L1 penalty
            # Threshold scale factor accounts for learning rate and lambda
            threshold = self._alpha * self._lambda
            self._W = self._soft_threshold(w_temp, threshold)
            
            # Update bias normally (bias is not regularized)
            self._b = self._b - self._alpha * db

        print("Lasso training completed")

    def get_weights(self):
        print(f"Weights = \n{self._W}")
        print(f"Bias = \n{self._b}")

    def predict(self, X):
        """Make predictions"""
        return X @ self._W + self._b 


if __name__ == "__main__":
    np.random.seed(45)
    data = np.random.rand(3000, 40)
    X = data[:, :-1]
    y = data[:, -1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train Custom Lasso
    lasso_custom = LassoRegularizer()
    lasso_custom.fit(X, y, alpha=0.01, lam=0.1, iterations=5000)
    print("____________________________________")
    print("FROM CUSTOM LASSO REGRESSION")
    print("____________________________________")
    lasso_custom.get_weights()

    # Compare with Scikit-Learn Lasso
    print("____________________________________")
    print("FROM SKLEARN LASSO")
    print("____________________________________")
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X, y)
    print(f"Intercept (Bias): {lasso_model.intercept_:.4f}")
    print(f"Coefficients (Weights): {lasso_model.coef_}")