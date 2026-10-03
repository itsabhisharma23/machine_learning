import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

class ElasticNetRegularizer:
    """Class to implement Elastic Net (L1 + L2) Regularization using Proximal Gradient Descent

    Objective (same as sklearn):
        (1 / 2m) * ||y - Xw - b||^2
        + lam * l1_ratio * ||w||_1
        + (lam * (1 - l1_ratio) / 2) * ||w||^2
    """

    def __init__(self):
        self._lambda = None
        self._alpha = None
        self._l1_ratio = None
        self._W = None
        self._b = None

    def _soft_threshold(self, w, thresh):
        """Soft-thresholding operator for L1 regularization"""
        return np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)

    def fit(self, X, y, alpha, lam, l1_ratio=0.5, iterations=1000):
        """Fit the data using Proximal Gradient Descent

        alpha    : learning rate
        lam      : overall regularization strength
        l1_ratio : mix between L1 and L2 (1.0 = Lasso, 0.0 = Ridge)
        """
        m, n_features = X.shape
        self._lambda = lam
        self._alpha = alpha
        self._l1_ratio = l1_ratio

        l1 = self._lambda * self._l1_ratio
        l2 = self._lambda * (1 - self._l1_ratio)

        print("Gradient Descent Implementation of Elastic Net Regression")

        # Initialize weights and bias
        self._W = np.zeros(n_features)
        self._b = 0.0

        for i in range(iterations):
            y_h = self.predict(X)

            # Cost calculation (MSE + L1 penalty + L2 penalty)
            cost = (np.sum((y_h - y)**2) / (2 * m)
                    + l1 * np.sum(np.abs(self._W))
                    + (l2 / 2) * np.sum(self._W**2))

            if i % (iterations // 5 if iterations >= 5 else 1) == 0:
                print(f"Cost after {i} iterations is {cost:06.4f}")

            # Standard gradient for MSE part
            dW = np.dot(X.T, (y_h - y)) / m
            db = np.sum(y_h - y) / m

            # Gradient step for weights (without penalty terms yet)
            w_temp = self._W - self._alpha * dW

            # Proximal operator for L1 + L2 penalty:
            # soft-threshold for the L1 part, then shrink for the L2 part
            self._W = self._soft_threshold(w_temp, self._alpha * l1) / (1 + self._alpha * l2)

            # Update bias normally (bias is not regularized)
            self._b = self._b - self._alpha * db

        print("Elastic Net training completed")

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

    # Train Custom Elastic Net
    enet_custom = ElasticNetRegularizer()
    enet_custom.fit(X, y, alpha=0.01, lam=0.01, l1_ratio=0.5, iterations=5000)
    print("____________________________________")
    print("FROM CUSTOM ELASTIC NET REGRESSION")
    print("____________________________________")
    enet_custom.get_weights()

    # Compare with Scikit-Learn ElasticNet
    print("____________________________________")
    print("FROM SKLEARN ELASTIC NET")
    print("____________________________________")
    enet_model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    enet_model.fit(X, y)
    print(f"Intercept (Bias): {enet_model.intercept_:.4f}")
    print(f"Coefficients (Weights): {enet_model.coef_}")
