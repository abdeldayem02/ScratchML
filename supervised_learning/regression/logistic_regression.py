import numpy as np

class LogisticRegression():
    """
    Logistic Regression Model:
    ----------------------------------------------------------------------------
    Logistic Regression is a classification algorithm that models the probability
    of a binary outcome using the logistic function (sigmoid). It can optionally
    include regularization to prevent overfitting.
    """

    def __init__(self, lambda_=0.0, regularization=None):
        """
        Initializes the Logistic Regression model.

        Parameters:
        - lambda_ (float): Regularization strength. Default is 0 (no regularization).
        - regularization (str): Type of regularization ('l1', 'l2', or None). Default is None.
        """
        self.theta = np.array([])
        self.regularization = regularization
        self.lambda_ = lambda_

    def _add_bias_term(self, X):
        """
        Adds a bias term (intercept) to the input data matrix X.

        Parameters:
        - X (array-like): Input feature matrix.

        Returns:
        - X (array-like): Feature matrix with an added column of ones.
        """
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        if not np.all(X[:,0] == 1):
            X = np.column_stack((np.ones(X.shape[0]),X))
    
        return X

    def _sigmoid(self, z):
        """
        Computes the sigmoid function for the given input.

        Parameters:
        - z (array-like): Input values.

        Returns:
        - (array-like): Sigmoid-transformed values.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, method="gd", learning_rate=0.001, epochs=1000):
        """
        Fits the Logistic Regression model using Gradient Descent with optional regularization.

        Parameters:
        - X (array-like): Input feature matrix.
        - y (array-like): Target labels (binary: 0 or 1).
        - method (str): Optimization method ('gd' for gradient descent). Default is 'gd'.
        - learning_rate (float): Learning rate for gradient descent. Default is 0.001.
        - epochs (int): Number of iterations for gradient descent. Default is 1000.
        """
        X = self._add_bias_term(X)
        self.theta = np.zeros(X.shape[1])  # Initialize weights to zeros

        if method == "gd":
            for _ in range(epochs):
                predictions = self._sigmoid(X @ self.theta)
                gradient = (1 / X.shape[0]) * (X.T @ (predictions - y))

                # Apply L2 Regularization
                if self.regularization == "l2":
                    gradient[1:] += self.lambda_ * self.theta[1:]

                # Apply L1 Regularization
                elif self.regularization == "l1":
                    gradient[1:] += self.lambda_ * np.sign(self.theta[1:])

                # Update weights
                self.theta -= learning_rate * gradient
        else:
            raise ValueError("Invalid method. Use 'gd' for gradient descent")

    def predict_proba(self, X):
        """
        Predicts probabilities for the input data.

        Parameters:
        - X (array-like): Input feature matrix.

        Returns:
        - (array-like): Predicted probabilities for each sample.
        """
        if len(self.theta) == 0:
            raise ValueError("Model is not trained yet. Call 'fit' before 'predict_proba'.")
        X = self._add_bias_term(X)
        return self._sigmoid(X @ self.theta)

    def predict(self, X, threshold=0.5):
        """
        Predicts binary class labels for the input data.

        Parameters:
        - X (array-like): Input feature matrix.
        - threshold (float): Decision threshold for classification. Default is 0.5.

        Returns:
        - (array-like): Predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(self, X, y):
        """
        Evaluate the accuracy of the model on the test data.

        Parameters:
        - X (array-like): Test feature data
        - y (array-like): True labels for the test data

        Returns:
        - float: Accuracy of the model (proportion of correct predictions)
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

