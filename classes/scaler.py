import numpy as np

class Scaler:
    """
        Scaler class used to apply Z-Score normalization to data.
        Each value is converted by using the formula:
            X_scaled = (X_original - mean of data) / standard deviation of data
    """
    def __init__(self):
        """
        Initialization of Scaler class.
        Mean and standard deviation set to None.
        """
        self.mean = None
        self.std = None

    def fit(self, X):
        """
        Compute mean and standard deviation.
        :param X: feature set.
        :return: void.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, ddof=0)

    def transform(self, X):
        """
        Apply Z-Score normalization to data.
        :param X: feature vector.
        :return: scaled feature vector.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            X_scaled = (X - self.mean) / np.where(self.std == 0, 1, self.std)
        return X_scaled

    def fit_transform(self, X):
        """
        Apply both fit and transform functions.
        :param X: feature set.
        :return: scaled feature set.
        """
        self.fit(X)
        return self.transform(X)