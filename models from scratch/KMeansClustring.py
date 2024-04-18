import numpy as np


def euclidean_distance(X1, X2):
    return np.sqrt(np.sum((X1 - X2) ** 2))

class KMeans:
    def __init__(self, n_clusters = 3, max_iters = 100,tol = 1e4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        self.tol = tol
        
    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        #initial valure for centroids
        random_idxs = np.random.choice(self.n_samples, self.n_clusters, replace=False)
        self.centroids = [self.X[idx] for idx in random_idxs]
        
        for _ in range(self.max_iters):
            #Assign samples to nearest centroid
            self.labels = self._create_clusters(self.centroids)
    
            #update centroids
            old_centroids = self.centroids
            self.centroids = self._update_centroids(self.labels)
            
            #cheack convergance
            if self._is_converged(old_centroids, self.centroids):
                break
          
            
    def _create_clusters(self, centroids, X = None):
        clusters = [[] for _ in range(self.n_clusters)]
        #Assign samples to nearest centroid
        for idx, value in enumerate(X if X is not None else self.X):
            centroid_idx = self._nearest_centroid(value, centroids)
            clusters[centroid_idx].append(idx)
            
        return clusters
    
    
    def _nearest_centroid(self,value, centroids):
        distances = [euclidean_distance(value, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
        
    
    
    def _update_centroids(self, labels):
        centroids = np.zeros((self.n_clusters, self.n_features))
        for idx, cluster in enumerate(labels):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean
        return centroids
    
    
    def _is_converged(self,old, new ):
        distances = [euclidean_distance(old[i], new[i]) for i in range(self.n_clusters)]
        return np.sum(distances) == self.tol

    def predict(self, X):
        predicted_label = []
        for i in X:
            idx = self._nearest_centroid(X,self.centroids)
            predicted_label.append(idx)
        return predicted_label
        