import numpy as np

class K_means():
    def __init__(self):
        pass

    def euclidean_distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
        return np.sqrt(distance)

    def centroids_init(self, k, X):
        n_samples, n_features = X.shape
        centroids = np.zeros((k, n_features))
        for i in range(k):
            # random choose a center point in each iteration
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def closest_centroid(self, sample, centroids):
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            # choose the class that the closet center point belongs to 
            distance = self.euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    def create_clusters(self, centroids, k, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(k)]
        for sample_i, sample in enumerate(X):
            # assign the sample to the closet class
            centroid_i = self.closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def calculate_centroids(self, clusters, k, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((k, n_features))
        # use the mean of all samples as the new center point
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def kmeans(self, X, k, max_iterations):
        centroids = self.centroids_init(k, X)
        for _ in range(max_iterations): 
            clusters = self.create_clusters(centroids, k, X) 
            prev_centroids = centroids
            centroids = self.calculate_centroids(clusters, k, X)
            diff = centroids - prev_centroids
            if not diff.any():
                break
                
        return self.get_cluster_labels(clusters, X)

if __name__=="__main__":
    k = K_means()
    X = np.array([[0,2],[0,0],[1,0],[5,0],[5,2]])
    labels = k.kmeans(X, 2, 10)
    print(labels)