import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

class RidgeRegularizer:
    """Class to implement Ridge or L2 Regularization"""

    def __init__(self):
        self._lambda = None
        self._alpha = None
        self._W = None
        self._b = None

    def fit(self,X,y,alpha,lam,iterations = 0):
        """ Function to fir Ridge regularization on the data"""
        m = X.shape[0]
        self._lambda = lam
        self._alpha = alpha
        if iterations == 0:
            print("Normal Form Implementation of Ridge Regression")
            X_b = np.hstack((np.ones((X.shape[0], 1)), X))
            # for ridge normal form equation will also have lambda.I
            penality = lam * np.eye(X_b.shape[1])
            # Do not regularize bias term
            penality[0,0] = 0
            self._W = np.linalg.inv(X_b.T @ X_b + penality) @ X_b.T @ y
            self._b = self._W[0]
            self._W = self._W[1:]
            print("Normal form training completed")

        else:
            print("Gradient Discent Implementation of Ridge Regression")
            self._W = np.zeros(X.shape[1])
            self._b = 0.0
            for _ in range(iterations):
                y_h = self.predict(X,y)
                cost = np.sum((y_h - y)**2)/(2*m) + (self._lambda / (2 * m)) * np.sum(self._W**2)
                if _ % 100 == 0:
                    # add no. of iterations and cost to cost_vector
                    print(f"Cost after {_} iterations is {cost:06.4f}")
                # Calculate Gradients
                dW = (np.dot(X.T,(y_h - y)) + self._lambda * self._W)/m
                db = np.sum(y_h - y)/m

                self._W = self._W - self._alpha * dW
                self._b = self._b - self._alpha * db

    def get_weights(self):
        print(f"Weights = \n{self._W}")
        print(f"Bias = \n{self._b}")

    def predict(self,X,y):
        """Make predictions"""
        return X @ self._W + self._b 
    


# Imitate training with the custom code here

if __name__ == "__main__":

    # Generate dummy data using numpy random

    np.random.seed(45)
    data = np.random.rand(3000,40)
    X = data[:,:-1]
    y = data[:,-1]

    #############################################
    # Ridge REGRESSION USING CUSTOM CODE (Gradient Descent)
    #############################################
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    ridge = RidgeRegularizer()
    ridge.fit(X,y,0.001,1)
    print("____________________________________")
    print("FROM NORMAL EQUATION")
    print("____________________________________")

    ridge.get_weights()

    print("____________________________________")
    print("FROM SKLEARN RIDGE")
    print("____________________________________")

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, y)

    print(f"Intercept (Bias): {ridge_model.intercept_:.4f}")
    print(f"Coefficients (Weights): {ridge_model.coef_}")


    ridge = RidgeRegularizer()
    ridge.fit(X,y,0.001,1,10000)
    print("____________________________________")
    print("FROM GRADIENT DISCENT")
    print("____________________________________")

    ridge.get_weights()






         

    
    

