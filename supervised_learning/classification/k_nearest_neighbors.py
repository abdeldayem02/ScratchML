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

    def _compute_distances(self, X_test):
        """
        Compute distances between test points and all training points in a vectorized manner.
        
        Parameters:
        - X_test: Test feature data
        
        Returns:
        - Matrix of distances (shape: [n_test_samples, n_train_samples])
        """
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((X_test[:, np.newaxis, :] - self.X_train) ** 2, axis=2))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(X_test[:, np.newaxis, :] - self.X_train), axis=2)
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

    def _get_neighbors(self, distances):
        """
        Find the k nearest neighbors for all test points.
        
        Parameters:
        - distances: Matrix of distances (shape: [n_test_samples, n_train_samples])
        
        Returns:
        - Array of k nearest neighbor labels for each test point
        """
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        return self.y_train[k_indices]

    def predict(self, X_test):
        """
        Predict the labels for the given test data.
        
        Parameters:
        - X_test: Test feature data
        
        Returns:
        - Predicted labels for the test data
        """
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("Model has not been fitted yet. Call fit() before predict().")
        X_test = np.array(X_test)
        distances = self._compute_distances(X_test)
        neighbors = self._get_neighbors(distances)
        
        predictions = []
        for neighbor_labels in neighbors:
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            max_count = np.max(counts)
            prediction = unique[counts == max_count].min()
            predictions.append(prediction)
        
        return np.array(predictions)

    def predict_proba(self, X_test):
        """
        Predict the probability estimates for the given test data.
        
        Parameters:
        - X_test: Test feature data
        
        Returns:
        - Probability estimates for each class (shape: [n_test_samples, n_classes])
        """
        X_test = np.array(X_test)
        distances = self._compute_distances(X_test)
        neighbors = self._get_neighbors(distances)
        
        classes = np.unique(self.y_train)
        probabilities = []
        for neighbor_labels in neighbors:
            counts = np.array([np.sum(neighbor_labels == c) for c in classes])
            probabilities.append(counts / self.k)
        
        return np.array(probabilities)

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
        print(f"Accuracy: {accuracy:.4f}")  # Add this
        return accuracy
