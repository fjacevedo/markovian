import numpy as np
import sklearn.cluster as cluster
import sklearn.neighbors as neighbors
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class ClusterRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, n_clusters=5, n_neighbors=5, **kwargs):
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.regressor = regressor
        self.classifier = neighbors.KNeighborsClassifier(
            n_neighbors=self.n_neighbors, **kwargs)

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)

        cluster_ = cluster.AgglomerativeClustering(
            n_clusters=self.n_clusters)
        labels = cluster_.fit_predict(X)
        del cluster_

        self.classifier.fit(X, labels)
        self.KMeansRegressors = np.empty(
            shape=(self.n_clusters,), dtype='object')

        for k in range(self.n_clusters):
            idx = np.argwhere(labels == k).flatten()
            X_train = X[idx]
            y_train = y[idx]
            rg = clone(self.regressor)
            rg.fit(X_train, y_train)
            self.KMeansRegressors[k] = rg

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        labels = self.classifier.predict(X)
        preds = list()
        for k in np.unique(labels):
            idx = np.argwhere(labels == k).flatten()
            X_test = X[idx]
            shape = (1, -1) if X_test.shape[0] == 1 else None
            preds.append(
                self.KMeansRegressors[k].predict(X_test.reshape(shape)))
        y = np.vstack(preds)
        return y
