# Lib imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler as sklearnStandardScalar

# Standard Scaling using z - scores
class StandardScaler:
    """
    Takes a singlew feature vector or set of feature vectors and scale them with z-scores
    """

    def __init__(self):
        self.means = None
        self.stddevs = None
        
    
    def fit(self,X):
        X = np.array(X)
        self.means = np.mean(X,axis=0)
        self.stddevs = np.std(X,axis=0)

        self.stddevs[self.stddevs==0] = 1.0

    def transform(self,X):
        if self.means is None or self.stddevs is None:
            raise ValueError("Model was never fit on the data.")
        X = np.array(X)
        scaled_X = (X - self.means)/self.stddevs
        return scaled_X
    
    def fit_transform(self,X):
        self.fit(X)
        scaled_X = self.transform(X)
        return scaled_X





# Linear Regression Implementation
class LinearRegressor:
    """
    Linear Regression class to train and infer predictions
    """
    def __init__(self):
        self.W = None
        self.b = None
        self.cost_vector = []

    def _loss(self,y_h,y):
        """Function to calculate loss"""
        return (y - y_h)**2

    def _cost(self,y_h,y):
        m = y.shape[0]
        cost = np.sum(self._loss(y_h,y))/(2*m)
        return cost

    def _calculate_gradients(self,X,y_h,y):
        """Calculate gradients for the provided data"""
        m = X.shape[0]
        
        # calculate grads
        dw = np.dot(X.T,(y_h - y))/m
        db = np.sum(y_h - y)/m

        return dw,db

    def _gradient_descent(self,X,y,alpha=0.001,iterations=10000):
        self.W = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(iterations):
            y_h = self.predict(X)
            cost = self._cost(y_h,y)
            if _ % 100 == 0:
                # add no. of iterations and cost to cost_vector
                self.cost_vector.append((_,cost))
                print(f"Cost after {_} iterations is {cost:06.4f}")
            dw,db = self._calculate_gradients(X,y_h,y)
            self.W = self.W - (alpha * dw)
            self.b = self.b - (alpha * db)




    def fit(self,X,y,alpha,iterations):
        print(f"Starting model training with {iterations} iterations.")
        self._gradient_descent(X,y,alpha,iterations)
        print("Model training completed!")

    def predict(self,X):
        return np.dot(X,self.W) + self.b

    

# Imitate training with the custom code here

if __name__ == "__main__":

    # Generate dummy data using numpy random

    np.random.seed(45)
    data = np.random.rand(3000,40)
    X = data[:,:-1]
    y = data[:,-1]

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    lr = LinearRegressor()
    lr.fit(X,y,0.001,10000)


    # Linear Regression using scikit learn

    scaler = sklearnStandardScalar()
    # Fit on training data only and transform it
    X_train_scaled = scaler.fit_transform(X)

    # 4. Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y)

    # 6. View model parameters
    print("____________________________________")
    print("FROM SKLEARN TRAINING")
    print("____________________________________")
    print("Model Coefficients:", model.coef_)
    print("Model Intercept:", model.intercept_)

    print("____________________________________")
    print("FROM CUSTOM LR TRAINING")
    print("____________________________________")
    print(lr.W)
    print(lr.b)

    

