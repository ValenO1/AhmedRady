import numpy as np

class GradientDescentRegressor:
    def __init__(self, learning_rate = .01, epochs = 1000, type = "batch", batch_size = 20, penalty = None, alpha = .1, l1_ratio = .5, random_state= None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.type = type
        self.batch_size = batch_size
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.weights = None
        np.random.seed(random_state)
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)
        bias = np.ones(n_samples)
        X = np.c_[bias, X]
        for epoch in range(self.epochs):
            if self.type == "batch":
                gradient = self._compute_gradient(X, y)
            elif self.type == "mini batch":
                indices = np.random.choice(n_samples, self.batch_size, replace=False)
                gradient = self._compute_gradient(X[indices], y[indices])
            elif self.type == "stochastic":
                index = np.random.choice(n_samples)
                gradient = self._compute_gradient(X[[index]], y[[index]])
            else:
                raise TypeError("only batch, mini batch and stochastic are supported")

            self.weights -= self.learning_rate * 1 / (2*n_samples) * gradient
        self.y_mean = np.mean(y)
            
            
    def _compute_gradient(self, X, y):
        gradient = -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(self.weights)
        if self.penalty is not None:
            if self.penalty =="l1":
                penalty = self.alpha * np.sign(self.weights)
            elif self.penalty == "l2":
                penalty = 2 * self.alpha * self.weights
            elif self.penalty == "elastic net":
                l1_penalty = self.l1_ratio * self.alpha * np.sign(self.weights)
                l2_penalty = (1 - self.l1_ratio) * self.alpha * self.weights
                penalty = self.alpha * (l1_penalty + l2_penalty)
            else:
                raise ValueError("penalty can be None, l1, l2 or elastic net")
            
            gradient[1:] += penalty[1:]
        return gradient
    
    
    def predict(self, X):
        bias = np.ones(X.shape[0])
        X = np.c_[bias, X]
        return X.dot(self.weights)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        sst = np.sum((y - self.y_mean)**2)
        sse = np.sum((y - y_pred)**2) 
        r_square = 1 - (sse / sst)
        return r_square