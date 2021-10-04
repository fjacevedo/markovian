# !/usr/bin/env python
# coding: utf-8
import warnings
import numpy as np
from pathlib import Path
import sklearn.base as base
from multiprocessing import cpu_count
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_array, check_is_fitted

CPUs = cpu_count()
warnings.simplefilter('ignore')
CWD = Path.cwd()


class Markovian(base.BaseEstimator):
    def __init__(self, L=4, regressor=KNeighborsRegressor, **kwargs):
        self.L = L
        self.regressor = regressor(**kwargs)
        self.models = list([
            base.clone(self.regressor) for _ in range(self.L)])

    def __group_by_steps(self, X):
        # Group the data by time step in order to fit the L regressors
        steps, n_pts = X.shape
        days = steps//self.L
        out = np.zeros((self.L+1, days-1, n_pts))
        out[:-1] = np.array(
            [[X[d+i] for d in range(days-1)] for i in range(self.L)])
        out[-1] = X[1:days]
        return out

    def fit(self, X, y=None):
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

        _X = self.__group_by_steps(X)
        for i, model in enumerate(self.models):
            model.fit(X=_X[i], y=_X[i+1])

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X, current=0, steps=1):
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
        _X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        for step in range(steps):
            y = self.models[(current+step)%self.L].predict(_X)
            _X = y.copy()
        return y
