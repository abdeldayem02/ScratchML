import numpy as np

# NaiveBayes is a simple probabilistic classifier based on Bayes' theorem.
class NaiveBayes():

    def __init__(self):
        """
        Initialize the NaiveBayes classifier.
        """
        self.class_priors = {}  # Dictionary to store prior probabilities for each class
        self.class_stats = {}  # Dictionary to store mean and variance for each class and feature
        self.classes = None  # Array to store unique class labels

    def fit(self, X, y):
        """
        Fit the NaiveBayes model to the training data.

        Parameters:
        - X: Training feature data (shape: [n_samples, n_features])
        - y: Training labels (shape: [n_samples])
        """
        self.classes = np.unique(y)  # Identify unique class labels

        # Calculate prior probabilities for each class
        self.class_priors = {
            c: np.sum(y == c) / len(y) for c in self.classes
        }

        # Calculate mean and variance for each feature in each class
        for c in self.classes:
            X_c = X[np.where(y == c)]  # Extract samples belonging to class c
            self.class_stats[c] = {
                "mean": np.mean(X_c, axis=0),  # Mean of features for class c
                "var": np.var(X_c, axis=0)    # Variance of features for class c
            }

    def _gaussian_pdf(self, x, mean, var):
        """
        Compute the Gaussian probability density function for a given value.

        Parameters:
        - x: Value for which to compute the probability
        - mean: Mean of the Gaussian distribution
        - var: Variance of the Gaussian distribution

        Returns:
        - Probability density value
        """
        eps = 1e-9  # Small value to prevent division by zero
        coeff = 1 / np.sqrt(2 * np.pi * (var + eps))  # Coefficient of the Gaussian formula
        exponent = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))  # Exponential term
        return coeff * exponent

    def predict(self, X):
        """
        Predict the class labels for the given test data.

        Parameters:
        - X: Test feature data (shape: [n_test_samples, n_features])

        Returns:
        - Predicted class labels for the test data (shape: [n_test_samples])
        """
        predictions = []
        for sample in X:
            posteriors = {}  # Dictionary to store posterior probabilities for each class
            for c in self.classes:
                # Start with the log of the class prior
                posterior = np.log(self.class_priors[c])
                # Add the log of the likelihood for each feature
                for feature_idx in range(len(sample)):
                    mean = self.class_stats[c]["mean"][feature_idx]
                    var = self.class_stats[c]["var"][feature_idx]
                    posterior += np.log(self._gaussian_pdf(sample[feature_idx], mean, var))
                posteriors[c] = posterior
            # Choose the class with the highest posterior probability
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict the probability estimates for the given test data.

        Parameters:
        - X: Test feature data (shape: [n_test_samples, n_features])

        Returns:
        - Probability estimates for each class (shape: [n_test_samples, n_classes])
        """
        probabilities = []
        for sample in X:
            posteriors = {}  # Dictionary to store posterior probabilities for each class
            for c in self.classes:
                # Start with the log of the class prior
                posterior = np.log(self.class_priors[c])
                # Add the log of the likelihood for each feature
                for feature_idx in range(len(sample)):
                    mean = self.class_stats[c]["mean"][feature_idx]
                    var = self.class_stats[c]["var"][feature_idx]
                    posterior += np.log(self._gaussian_pdf(sample[feature_idx], mean, var))
                posteriors[c] = np.exp(posterior)  # Convert log-posterior back to probability
            # Normalize probabilities
            total = sum(posteriors.values())
            probabilities.append([posteriors[c] / total for c in self.classes])
        return np.array(probabilities)

    def evaluate(self, X, y):
        """
        Evaluate the accuracy of the model on the test data.

        Parameters:
        - X: Test feature data (shape: [n_test_samples, n_features])
        - y: True labels for the test data (shape: [n_test_samples])

        Returns:
        - Accuracy of the model (float)
        """
        y_pred = self.predict(X)  # Predict class labels for the test data
        accuracy = np.sum(y_pred == y) / len(y)  # Calculate accuracy
        return accuracy




