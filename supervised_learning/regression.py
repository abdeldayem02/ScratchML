import numpy as np
from itertools import combinations_with_replacement

class Regression(object):
    """
    -----------------------------
    Bass Class for the regression models. Models where the relationship between
    y(target variable) and X(input features) where  y = Xθ + ϵ
    -----------------------------
    """
    def __init__(self):
        """
        theta: represents a vector of parameters (weights)
        """
        self.theta = None
        

    def _add_bias_term(self, X):
        """
        Regression models often require a bias term (intercept)
        This function ensures the input data matrix X always has a column of ones.
        """
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        if not np.all(X[:,0] == 1):
            X = np.column_stack((np.ones(X.shape[0]),X))
    
        return X
    
    def fit(self, X, y):
        """
        Each model will has its own fitting version,
        so this function (the same for predict() function) is a placeholder (empty).
        """
        raise NotImplementedError("Subclasses must implement the fit method.")
        
    
    def predict(self, X):
        
        raise NotImplementedError("Subclasses must implement the fit method.")
        

    def evaluate(self, X, y, metric = "mse"):
        """
        A generic method for evaluating model performance with multiple mertics like:
        
        - Mean Squared Error (MSE) which is the default
        - Mean Absolute Error (MAE)
        - R² Score
        """
        X = self._add_bias_term(X)
        y_pred = self.predict(X)

        if metric == "mse":
            mse = np.mean((y-y_pred)**2)
            print(f"MSE: {mse}")
        elif metric == "mae":
            mae = np.mean(np.abs(y - y_pred))
            print(f"MAE: {mae}")
        elif metric == "r2":
            r2 = 1 - (np.sum((y-y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            print(f"R2: {r2}")
        else:
            raise ValueError("Unsupported metric. Use 'mse', 'mae', or 'r2'")
        

class LinearRegression(Regression):
    """
    I used 2 methods for the Linear Regression:
    ----------------------------------------------------------------------------
    1- **Normal Equation**: provides a closed-form solution to Linear Regresssion, meaning it calcuates
    the optimal weights (θ) directly without needing iteration (as in Gradient Descent).
    The Formula for the Normal Equation is: θ = inv((X.T * X)) * X.T * y 
    where inv means the inverse and T means is the transpose
    
    2- **Gradient Descent**: is an optimization algorithm used to find the optimal parameters (weights)
    for a machine learning model by minimizing a cost function, and here in Linear Regression,
    we use it to minimize the Mean Squared Error (MSE) between predicted and actual values.

    """
    def __init__(self):
        super().__init__()

    def fit(self, X, y, method = "normal", learning_rate=0.001, epochs=1000):
        X = self._add_bias_term(X)
        y = np.array(y)
        if method == "normal":
            self.theta = np.linalg.inv(X.T @ X) @ X.T @ y
            
        elif method == "gd":
            self.theta = np.zeros(X.shape[1])
            for _ in range(epochs):
                gradient = (1 / X.shape[0]) * (X.T @ (X @ self.theta - y))
                self.theta -= learning_rate * gradient

        else:
            raise ValueError("Invalid method. Use 'normal' or 'gd'. ")
    
    def predict(self, X):
        X = self._add_bias_term(X)
        y_pred = np.dot(X,self.theta)
        return y_pred
    
    def evaluate(self, X, y, metric="mse"):
        return super().evaluate(X, y, metric)
    
            

class RidgeRegression(Regression):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def fit(self, X, y, method="normal", learning_rate=0.001, epochs=1000):
        X = self._add_bias_term(X)

        if method == "normal":
            I = np.eye(X.shape[1])  # Identity matrix
            I[0, 0] = 0  # Don't penalize the bias term
            self.theta = np.linalg.inv(X.T @ X + self.lambda_ * I) @ X.T @ y

        elif method == "gd":
            self.theta = np.zeros(X.shape[1])
            for _ in range(epochs):
                gradient = (1 / X.shape[0]) * (X.T @ (X @ self.theta - y) + self.lambda_ * self.theta)
                self.theta -= learning_rate * gradient

        else:
            raise ValueError("Invalid method. Use 'normal' or 'gd'.")

    def predict(self, X):
        X = self._add_bias_term(X)
        return np.dot(X, self.theta)

class LassoRegression(Regression):
    
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def fit(self, X, y, learning_rate = 0.001,epochs = 1000):
        X = self._add_bias_term(X)
        self.theta = np.zeros(X.shape[1])
        for _ in range(epochs):
            gradient = (1/X.shape[0]) * (X.T @ (X @ self.theta - y))
            gradient[1:] += self.lambda_ * np.sign(self.theta[1:])
            
            self.theta -= learning_rate * gradient

    def predict(self, X):
        X = self._add_bias_term(X)
        return X @ self.theta

class PolynomialRegression(Regression):  

    def __init__(self, degree = 2):
        super().__init__()
        self.degree = degree

    def _transform_features(self, X):
        n_samples, n_features = X.shape

        X_transformed = []

        for i in range(n_features):
            X_transformed.append(X[:, i].reshape(-1,1))
        
        for deg in range(2, self.degree + 1):
            for combination in combinations_with_replacement(range(n_features), deg):
                term = np.prod(X[:, combination], axis=1).reshape(-1,1)
                X_transformed.append(term)

            
        return np.hstack(X_transformed)
    
    def fit(self, X, y, method = "normal", learning_rate = 0.001, epochs = 1000):
        X_poly = self._transform_features(X)
        X_poly = self._add_bias_term(X_poly)

        if method == "normal":
            self.theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
        elif method == "gd":
            self.theta = np.zeros(X_poly.shape[1])
            for _ in range(epochs):
                gradient = (1 / X_poly.shape[0]) * (X_poly.T @ (X_poly @ self.theta - y))
                self.theta -= learning_rate * gradient

    
    def predict(self, X):
        
        X_poly = self._transform_features(X)
        X_poly = self._add_bias_term(X_poly)
        predictions = X_poly @ self.theta
        return predictions

        