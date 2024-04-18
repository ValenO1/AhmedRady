import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, iterations=1000, C=1, kernel="linear", degree=2, gamma=1.0):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.alpha = None
        self.w = None
        self.b = None
        self.X_train = None
        self.y_train = None

    def kernel_function(self, x1, x2):
        if self.kernel == "linear":
            return np.dot(x1, x2.T)
        elif self.kernel == "poly":
            return (np.dot(x1, x2.T) + 1) ** self.degree
        elif self.kernel == "rbf":
            norm = np.linalg.norm(x1[:, np.newaxis] - x2, axis=2)
            return np.exp(-self.gamma * norm ** 2)
        else:
            raise ValueError("Unsupported kernel type. Supported types: 'linear', 'poly', 'rbf'")

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.alpha = np.zeros(num_samples)
        self.X_train = X
        self.y_train = y

        # Kernel matrix computation
        K = self.kernel_function(X, X)

        # Gradient ascent for alpha optimization
        for _ in range(self.iterations):
            gradient = 1 - (y * K.dot(self.alpha * y))
            self.alpha += self.learning_rate * gradient
            self.alpha = np.clip(self.alpha, 0, self.C)  # Enforce 0 <= alpha <= C for soft margin

        # Calculate the weight vector (w) and bias (b) for linear kernel only
        if self.kernel == "linear":
            self.w = np.sum((self.alpha * y)[:, np.newaxis] * X, axis=0)
            self.b = np.mean(y - np.dot(X, self.w))

    def decision_function(self, X):
        if self.kernel == "linear":
            return np.dot(X, self.w) + self.b
        else:
            K = self.kernel_function(X, self.X_train)
            return np.dot(K, self.alpha * self.y_train) + self.b

    def predict(self, X):
        decision_values = self.decision_function(X)
        return np.where(decision_values >= 0, 1, -1)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)