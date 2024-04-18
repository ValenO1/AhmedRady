import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = .01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for i in range(self.iterations):
            y_pred = self.predict_proba(X)
            
            d_weights = (-1 / n_samples) * X.T.dot(y - y_pred)
            d_bias = (-1 / n_samples) * np.sum(y - y_pred)
            
            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias
            
    
    def predict_proba(self, X):
        z = X.dot(self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X):
        predictions = self.predict_proba(X)
        return np.where(predictions >=0.5, 1, 0)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred) / len(y)
        return accuracy