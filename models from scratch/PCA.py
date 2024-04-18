import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self,X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov_matrix = np.cov(X.T)
        eign_values, eign_vectors = np.linalg.eig(cov_matrix)

        idxs = np.argsort(eign_values)[::-1]
        eign_values = eign_values[idxs]
        eign_vectors = eign_vectors[idxs]
        self.components = eign_vectors[:, :self.n_components]

    def transform(self, X):
        X = X - self.mean
        return X.dot(self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        self.transform(X)