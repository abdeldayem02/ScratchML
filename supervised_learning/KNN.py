import numpy as np

class KNN():
    """
    K-Nearest Neighbors (KNN) implementation from scratch.
    
    Parameters:
    - k: Number of neighbors to consider (default: 3)
    - distance_metric: Distance metric to use ('euclidean' or 'manhattan', default: 'euclidean')
    """
    def __init__(self, k=3, distance_metric="euclidean"):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Parameter 'k' must be a positive integer.")
        self.k = k
        if distance_metric not in ["euclidean", "manhattan"]:
            raise ValueError("Unsupported distance metric. Choose 'euclidean' or 'manhattan'.")
        self.distance_metric = distance_metric

    def _compute_distance(self, X1, X2):
        """
        Compute the distance between two points based on the selected distance metric.
        
        Parameters:
        - X1: First data point
        - X2: Second data point
        
        Returns:
        - Distance between X1 and X2
        """
        if self.distance_metric == "euclidean":
            # Euclidean distance formula
            return np.sqrt(np.sum((X1 - X2) ** 2))
        elif self.distance_metric == "manhattan":
            # Manhattan distance formula
            return np.sum(np.abs(X1 - X2))
        else:
            raise ValueError("Unsupported distance metric. Choose 'euclidean' or 'manhattan'.")

    def fit(self, X_train, y_train):
        """
        Store the training data for later use during prediction.
        
        Parameters:
        - X_train: Training feature data
        - y_train: Training labels
        """
        if len(X_train) < self.k:
            raise ValueError("Parameter 'k' cannot be greater than the number of training samples.")
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _get_neighbors(self, X_test, X_train, y_train):
        """
        Find the k nearest neighbors for a given test point.
        
        Parameters:
        - X_test: Test data point
        - X_train: Training feature data
        - y_train: Training labels
        
        Returns:
        - Labels of the k nearest neighbors
        """
        # Compute distances from the test point to all training points
        distances = [self._compute_distance(X_test, x_train) for x_train in X_train]
        
        # Get indices of the k smallest distances
        k_indices = np.argsort(distances)[:self.k]
        
        # Retrieve the labels of the k nearest neighbors
        neighbors = y_train[k_indices]
        return neighbors

    def predict(self, X_test):
        """
        Predict the labels for the given test data.
        
        Parameters:
        - X_test: Test feature data
        
        Returns:
        - Predicted labels for the test data
        """
        X_test = np.array(X_test)
        predictions = []
        
        for x in X_test:
            # Get the k nearest neighbors for the current test point
            neighbors = self._get_neighbors(x, self.X_train, self.y_train)
            
            # Perform majority voting to determine the predicted label
            unique, counts = np.unique(neighbors, return_counts=True)
            max_count = np.max(counts)
            # Handle ties by selecting the smallest label in case of a tie
            prediction = unique[counts == max_count].min()
            predictions.append(prediction)
        
        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the accuracy of the model on the test data.
        
        Parameters:
        - X_test: Test feature data
        - y_test: True labels for the test data
        
        Returns:
        - Accuracy of the model
        """
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        y_pred = self.predict(X_test)
        
        # Calculate accuracy as the proportion of correct predictions
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        return accuracy
