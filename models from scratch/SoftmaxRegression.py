import numpy as np
from sklearn.preprocessing import OneHotEncoder

class SoftmaxRegression:
    def __init__(self,epochs = 1000, learning_rate= .1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.biases = None
        
        
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, keepdims=True, axis=1))
        return exp_z / np.sum(exp_z, keepdims= True, axis= 1)
        
        
    def gradient_descent(self,X, y):
        for i in range(self.epochs):
            scores = np.dot(X, self.weights) + self.biases
            probs = self.softmax(scores)
            gradient_w = np.dot(X.T, (probs - y)) / self.num_samples
            gradient_b = np.mean((probs - y), axis=0)
            self.weights -= self.learning_rate * gradient_w
            self.biases -= self.learning_rate * gradient_b
    
    def fit(self, X, y):
        num_calsses = len(np.unique(y))
        self.num_samples, num_features = X.shape
        self.weights = np.zeros((num_features, num_calsses))
        self.biases = np.zeros(num_calsses)
        y = y.reshape(-1,1)
        encoder = OneHotEncoder()
        y_encoded = encoder.fit_transform(y).toarray()
        self.gradient_descent(X, y_encoded)
        
    def predict_proba(self, X):
        scores = np.dot(X, self.weights) + self.biases
        return self.softmax(scores)
    
    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis= 1)
    
    def score(self, X, y):
        predicted_label = self.predict(X)
        return np.mean(predicted_label == y)