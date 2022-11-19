"""
Semi - Supervised K - Means
"""
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from functools import reduce


class KMeans(object):

    def __init__(self, k, tol=0.0001, max_iter=30, metric='euclidean', known_data=None, alpha=0.5, seed=1234, verbose=False):
        """
            :param k: The number of clusters
            :param threshold: The convergance threshold (default: 0.0001)
            :param max_iter: The max number of iterations if convergance did not reach (default: 30)
            :param metric: The distance metric to use. Valid values are:
                "euclidean" (default), "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobis"
            :param known_data: A 2D array of indices of known points. When using this parameter, every cluster that
                does not contain known labels should be represented as empty list. For example, if we have a dataset and
                we know some points for classes 1 and 3, we would have
                ```known_data=[np.array([1,2,3,4]), np.array([]), np.array([19,20,21])]```
            :param alpha: When using semi supervised
                    clustering, we can weigh the known data points differently.
                    The range of this paremeter is between 0 <= alpha <= 1.
            :param verbose: Prints iterations and convergence rate when set to True (default: False)

            :type k: int
            :type tol: float
            :type max_iter: int
            :type metric: str
            :type know_data: list
            :type alpha: float
            :type verbose: bool
        """
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.labels_  = None
        self.metric = metric
        self.known_data = known_data
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed=seed)
        self.verbose = verbose

    def _validate_metric(self):
        """
            Validates that the metric parameter
        """
        try:
            pairwise_distances(np.array([0,0]).reshape(-1,1), np.array([1,1]).reshape(-1,1), metric=self.metric)
        except Exception as e:
            print(e)
            return

    def _get_distance(self, X, Y=None, norm_X=None):
        """
            X: [N_X, n_feat]
            Y: [N_Y, n_feat]
            norm_X: [N_X]
            returns: [N_X, N_Y or k]
        """
        if self.metric == 'cosine':
            norm_X = norm_X if norm_X is not None else np.einsum('ij,ji->i', X, X.T)  # (X @ X.T).diagnoal()
            norm_X = norm_X.reshape(-1, 1)  # [N_X, 1]
            Y = Y if Y is not None else self.centroids
            norm_Y = np.einsum('ij,ji->i', Y, Y.T).reshape(1, -1)  # [1, k]

            sim = X @ Y.T / (norm_X * norm_Y)**.5
            sim[(sim>1.)  & (abs(sim-1.)<1e-4)] = 1.
            sim[(sim<-1.) & (abs(sim+1.)<1e-4)] = -1.
            theta = np.arccos(sim)
            return theta
        else:
            return pairwise_distances(X, self.centroids, metric=self.metric)

    def _update_centroids(self, data):
        """
            Updates the clusters centroids by taking the mean of all the points the belongs ot that cluster
            :param data: The data to cluster
            :type data: np.array
        """
        for i in range(self.k):
            self.centroids[i] = np.mean(data[np.where(self.labels_==i)], axis=0)

    def _update_biased_centroids(self, X):
        """
            Updates the clusters centroids in the semi-supervised settings
            :param data: The data to cluster
            :type data: np.array
        """
        weights = (1-self.alpha) * np.ones_like(X)
        all_known_data = np.array(reduce(lambda x1, x2: np.hstack([x1, x2]), self.known_data, np.array([])), dtype=np.int64)
        weights[all_known_data] = self.alpha
        weights = weights / weights.sum()

        for i in range(self.k):
            # compute weights for every cluster
            inds = np.where(self.labels_==i)[0]
            self.centroids[i] = (weights[inds] * X[inds]).sum(axis=0)

    def _kmeans_pp(self, X):
        """
            KMeans++: Initialize the cluster centers

            :param X: The dataset to cluster
        """
        self.centroids = np.array([X[data_indices_for_c].mean(axis=0) for data_indices_for_c in self.known_data])

        centroid_args = []
        norm_X = np.einsum('ij,ji->i', X, X.T)
        new_centroid = self.centroids[-1]
        for c in range(self.k-len(self.centroids)):
            y = new_centroid.reshape(1, -1)  # [1, n_feat]
            d = self._get_distance(X, y, norm_X=norm_X).reshape(-1) ** .3  # [N]
            d[centroid_args] = 0
            new_centroid_arg = self.rng.choice(X.shape[0], p=d/d.sum())
            centroid_args.append(new_centroid_arg)
            new_centroid = X[new_centroid_arg]
            self.centroids = np.vstack([self.centroids, new_centroid])
        self.labels_ = self.predict(X)

    def predict_train(self):
        """
            :return: The labels of the clustered data
            :rtype: np.array
        """
        return self.labels_, self.centroids

    def fit_predict(self, data):
        """
            Clusteres the data and returns tha labels

            :param data: The data to cluster
            :type data: np.ndarray
            :return: The data labels
            :rtype: np.array
        """
        self.fit(data)
        return self.predict_train()

    def fit(self, X):
        """
            Clusters the data

            :param data: The data to cluster
            :type data: np.ndarray
        """
        # Find initial centroids
        self._kmeans_pp(X)
        norm_X = np.einsum('ij,ji->i', X, X.T)
        new_labels = self.labels_.copy()
        counter = 0
        diff = np.infty
        while counter < self.max_iter and self.tol < diff:
            new_labels = np.argmin(self._get_distance(X, norm_X=norm_X), axis=1).astype(np.int64)
            old_centroids = self.centroids.copy()
            self.labels_ = new_labels.copy()
            if self.known_data is not None:
                self._update_biased_centroids(X)
            else:
                self._update_centroids(X)
            diff = ((old_centroids - self.centroids) ** 2).mean() ** .5
            counter += 1
            if self.verbose:
                print(f"Iter {counter:3d}\tDiff: {diff:.4e}")
        return self

    def predict(self, X_, only_known=False):
        """
            :return: The labels of the data
            :rtype: np.array
        """
        distances = self._get_distance(X_)
        if only_known:
            distances = distances[:]
        new_labels = np.argmin(distances, axis=1).astype(np.int64)

        return new_labels
