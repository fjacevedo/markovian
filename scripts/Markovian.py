# !/usr/bin/env python
# coding: utf-8
import warnings
import itertools as itr
from pathlib import Path

import numpy as np
import numpy.random as rnd

import scipy.sparse as sparse
from scipy.sparse import linalg as splinalg

import sklearn.base as base
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_array, check_is_fitted

warnings.simplefilter('ignore')
CWD = Path.cwd()


class Markovian(base.BaseEstimator):
    def __init__(self, N_lat, N_lon, L=4, regressor=KNeighborsRegressor,
                 **kwargs):
        self.L = L
        self.N_LAT = N_lat
        self.N_LON = N_lon
        self.N_POINTS = N_lat*N_lon
        self.L_GRID = [
            np.arange(self.N_POINTS).reshape(self.N_LAT, self.N_LON)]
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
            y = self.models[(current+step) % self.L].predict(_X)
            _X = y.copy()
        return y

# DATA ASSIMILATION

    def __get_data_structure(self, r, N_var=1):
        P = []
        for v, y, x in itr.product(range(N_var), range(self.N_LAT),
                                   range(self.N_LON)):
            ran_lat = np.arange(max(0, y-r), min(self.N_LAT, y+r+1))
            ran_lon = np.arange(x-r, x+r+1) % self.N_LON
            ind_loc = np.array(
                [[j, i] for j in ran_lat for i in ran_lon])
            local_box = self.L_GRID[v][ind_loc[:, 0], ind_loc[:, 1]]
            local_box_add = np.array([
                self.L_GRID[z][ind_loc[:, 0], ind_loc[:, 1]]
                for z in range(0, v)])
            local_box_add = local_box_add.flatten()
            local_box = np.hstack(
                (local_box, local_box_add)).astype('int32')

            label_grid = self.L_GRID[v][y, x]
            local_pred = local_box[local_box < label_grid].astype('int32')

            P.append(local_pred)

        return P

    def __get_inv_ridge(self, P, X, alp=0.1):
        n = len(P)
        _I = list(range(n))
        J = list(range(n))
        LV = n*list([1])
        DV = np.zeros(n)
        DV[0] = 1/np.var(X[0, :])
        lr = linear_model.Ridge(alpha=alp, fit_intercept=False)
        for i in range(1, n):
            pred = P[i]
            N_pred = pred.size
            y = X[i, :]
            Z = X[pred, :].T
            Z = Z.reshape(-1, 1) if N_pred == 1 else Z
            lr.fit(Z, y)
            res = y-lr.predict(Z)

            _I.extend(N_pred*list([i]))
            J.extend(pred)
            LV.extend(-lr.coef_.flatten())

            DV[i] = 1/np.var(res)

        _I = np.array(_I).astype('int32')
        J = np.array(J).astype('int32')

        L = sparse.coo_matrix((LV, (_I, J)))
        D_inv = sparse.diags(DV)

        return L, D_inv

    def __get_H(self, idx, n_points):
        H = sparse.lil_matrix((idx.size, n_points))
        for i, j in enumerate(idx):
            H[i, j] = 1
        out = H.tocsc()
        return out

    def init_EnKFMC(self, std, atms_ens, r, p_obs):
        self._size = int(self.N_POINTS*p_obs)
        self._atms_ens = self.__group_by_steps(atms_ens)
        self._Pr = self.__get_data_structure(r)
        self._R = sparse.diags(np.array(1/std**2).repeat(self.N_POINTS))
        self._xb = self._atms_ens[0].T.mean(1).reshape(-1, 1)

    def step_EnKFMC(self, atms_obs, seed=None):
        rnd.seed(seed=seed)
        points = np.arange(self.N_POINTS)
        step = 0
        while True:
            Xb = self._atms_ens[step % self.L].T
            Obs_choosed = rnd.choice(points, size=self._size, replace=False)
            Obs_choosed.sort()
            DX = Xb-np.outer(self._xb, np.ones(Xb.shape[1]))
            L, D_inv = self.__get_inv_ridge(self._Pr, DX)
            Bk_ = L.T.dot(D_inv.dot(L))
            H = self.__get_H(Obs_choosed, self.N_POINTS)
            y = atms_obs[step].reshape(-1, 1)
            d = H.dot(y - self._xb)
            _Rk = H.dot(self._R.dot(H.T))
            Z = splinalg.spsolve(
                Bk_ + H.T.dot(_Rk.dot(H)),
                H.T.dot(_Rk.dot(d)))
            xa = self._xb + Z.reshape(-1, 1)
            yield xa.ravel()
            self._xb = self.predict(
                xa.reshape(1, -1), current=(step+1) % self.L)
            self._xb = self._xb.reshape(-1, 1)
            step += 1
