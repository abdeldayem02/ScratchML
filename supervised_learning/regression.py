from itertools import combinations_with_replacement
import numpy as np

class Regression(object):
    """
    -----------------------------
    Base Class for the regression models. Models where the relationship between
    y(target variable) and X(input features) is defined as y = Xθ + ϵ
    -----------------------------
    """
    def __init__(self):
        """
        Initializes the base Regression model.

        Attributes:
            theta (numpy.ndarray): Vector of parameters (weights) for the regression model.
        """
        self.theta = np.array([])  # Initialize as an empty array

    def _add_bias_term(self, X):
        """
        Regression models often require a bias term (intercept).
        This function ensures the input data matrix X always has a column of ones.
        """
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if not np.all(X[:, 0] == 1):
            X = np.column_stack((np.ones(X.shape[0]), X))
    
        return X

    def _fit_normal_equation(self, X, y, regularization=None):
        """
        Fits the model using the Normal Equation.

        Args:
            X (numpy.ndarray): Input features with bias term added.
            y (numpy.ndarray): Target variable.
            regularization (float or None): Regularization strength for L2 regularization.

        Returns:
            numpy.ndarray: Optimal weights (theta).
        """
        I = np.eye(X.shape[1])
        if regularization:
            I[0, 0] = 0  # Don't penalize the bias term
            return np.linalg.pinv(X.T @ X + regularization * I) @ X.T @ y
        return np.linalg.pinv(X.T @ X) @ X.T @ y

    def _gradient_descent(self, X, y, learning_rate, epochs, regularization=None, l1=False):
        """
        Performs gradient descent to optimize theta.

        Parameters
        ----------
        X : numpy.ndarray
            Input features with bias term added.
        y : numpy.ndarray
            Target variable.
        learning_rate : float
            Learning rate for gradient descent.
        epochs : int
            Number of iterations for gradient descent.
        regularization : float, optional
            Regularization strength for L2 regularization. Default is None.
        l1 : bool, optional
            Whether to apply L1 regularization (Lasso). Default is False.
        """
        self.theta = np.zeros(X.shape[1])
        for _ in range(epochs):
            gradient = (1 / X.shape[0]) * (X.T @ (X @ self.theta - y))
            if regularization:
                if l1:
                    gradient[1:] += regularization * np.sign(self.theta[1:])
                else:
                    gradient[1:] += regularization * self.theta[1:]
            self.theta -= learning_rate * gradient

    def predict(self, X):
        """
        Predicts the target variable for the given input data X using the learned parameters (theta).
        Subclasses can override this method if needed.

        Args:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted target values.
        """
        if self.theta is None:
            raise ValueError("Model is not trained yet. Call 'fit' before 'predict'.")
        
        X = self._add_bias_term(X)
        return np.dot(X, self.theta)

    def evaluate(self, X, y, metric="mse"):
        """
        A generic method for evaluating model performance with multiple metrics like:
        
        - Mean Squared Error (MSE) which is the default
        - Mean Absolute Error (MAE)
        - R² Score
        """
        X = self._add_bias_term(X)
        y_pred = self.predict(X)

        if metric == "mse":
            mse = np.mean((y - y_pred) ** 2)
            print(f"MSE: {mse}")
        elif metric == "mae":
            mae = np.mean(np.abs(y - y_pred))
            print(f"MAE: {mae}")
        elif metric == "r2":
            r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            print(f"R2: {r2}")
        else:
            raise ValueError("Unsupported metric. Use 'mse', 'mae', or 'r2'.")


class LinearRegression(Regression):
    """
    Linear Regression Model:
    ----------------------------------------------------------------------------
    1- **Normal Equation**: Provides a closed-form solution to Linear Regression,
    calculating the optimal weights (θ) directly without iteration.
    Formula: θ = inv((X.T * X)) * X.T * y 
    
    2- **Gradient Descent**: An optimization algorithm used to find the optimal
    parameters (weights) by minimizing the Mean Squared Error (MSE).
    """

    def fit(self, X, y, method="normal", learning_rate=0.001, epochs=1000):
        """
        Fits the Linear Regression model using the specified method:
        - 'normal': Uses the Normal Equation.
        - 'gd': Uses Gradient Descent.
        """
        X = self._add_bias_term(X)
        y = np.array(y)

        if method == "normal":
            self.theta = self._fit_normal_equation(X, y)

        elif method == "gd":
            self._gradient_descent(X, y, learning_rate, epochs)

        else:
            raise ValueError("Invalid method. Use 'normal' or 'gd'.")



class RidgeRegression(Regression):
    """
    Ridge Regression Model:
    ----------------------------------------------------------------------------
    Ridge Regression adds L2 regularization to the cost function to prevent
    overfitting by penalizing large weights.
    """
    def __init__(self, lambda_=1.0):
        """
        lambda_: Regularization strength. Higher values mean stronger regularization.
        """
        self.lambda_ = lambda_

    def fit(self, X, y, method="normal", learning_rate=0.001, epochs=1000):
        """
        Fits the Ridge Regression model using the specified method:
        - 'normal': Uses the Normal Equation with L2 regularization.
        - 'gd': Uses Gradient Descent with L2 regularization.
        """
        X = self._add_bias_term(X)

        if method == "normal":
            self.theta = self._fit_normal_equation(X, y, regularization=self.lambda_)

        elif method == "gd":
            self._gradient_descent(X, y, learning_rate, epochs, regularization=self.lambda_)

        else:
            raise ValueError("Invalid method. Use 'normal' or 'gd'.")



class LassoRegression(Regression):
    """
    Lasso Regression Model:
    ----------------------------------------------------------------------------
    Lasso Regression adds L1 regularization to the cost function to encourage
    sparsity in the model by driving some weights to zero.
    """
    def __init__(self, lambda_=1.0):
        """
        lambda_: Regularization strength. Higher values mean stronger regularization.
        """
        self.lambda_ = lambda_

    def fit(self, X, y, learning_rate = 0.001, epochs = 1000):
        """
        Fits the Lasso Regression model using Gradient Descent with L1 regularization.
        """
        X = self._add_bias_term(X)
        self._gradient_descent(X, y, learning_rate, epochs, regularization=self.lambda_, l1=True)



class PolynomialRegression(Regression):  
    """
    Polynomial Regression Model:
    ----------------------------------------------------------------------------
    Extends Linear Regression by transforming the input features into polynomial
    features of a specified degree.
    """
    def __init__(self, degree=2):
        """
        degree: The degree of the polynomial features to generate.
        """
        self.degree = degree

    def _transform_features(self, X):
        """
        Transforms the input features into polynomial features up to the specified degree.
        """
        _, n_features = X.shape

        X_transformed = []

        for i in range(n_features):
            X_transformed.append(X[:, i].reshape(-1, 1))
        
        for deg in range(2, self.degree + 1):
            for combination in combinations_with_replacement(range(n_features), deg):
                term = np.prod(X[:, combination], axis=1).reshape(-1, 1)
                X_transformed.append(term)

        return np.hstack(X_transformed)
    
    def fit(self, X, y, method="normal", learning_rate=0.001, epochs=1000):
        """
        Fits the Polynomial Regression model using the specified method:
        - 'normal': Uses the Normal Equation.
        - 'gd': Uses Gradient Descent.
        """
        X_poly = self._transform_features(X)
        X_poly = self._add_bias_term(X_poly)

        if method == "normal":
            self.theta = self._fit_normal_equation(X_poly, y)

        elif method == "gd":
            self._gradient_descent(X_poly, y, learning_rate, epochs)

    def predict(self, X):
        """
        Predicts the target variable for the given input data X.
        """
        X_poly = self._transform_features(X)
        X_poly = self._add_bias_term(X_poly)
        return X_poly @ self.theta

