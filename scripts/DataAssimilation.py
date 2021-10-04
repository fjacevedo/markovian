# !/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.random as rnd
import scipy.sparse as sparse
import itertools as itr

from sklearn import linear_model
from sklearn import neighbors
from scipy.sparse import linalg as splinalg

import warnings
warnings.simplefilter('ignore')


class Markovian3DVAR():
    def __init__(self, N_lat, N_lon) -> None:
        self.N_lat = N_lat
        self.N_lon = N_lon
        self.N_points = N_lat*N_lon
        self.L_grid = [
            np.arange(self.N_points).reshape(self.N_lat, self.N_lon)]

    def __get_data_structure(self, r, N_lat, N_lon, N_var, L_grid):
        P = []
        for v, y, x in itr.product(range(N_var), range(N_lat), range(N_lon)):
            ran_lat = np.arange(max(0, y-r), min(N_lat, y+r+1))
            ran_lon = np.arange(x-r, x+r+1) % N_lon
            ind_loc = np.array(
                [[j, i] for j in ran_lat for i in ran_lon])
            local_box = L_grid[v][ind_loc[:, 0], ind_loc[:, 1]]
            local_box_add = np.array([
                L_grid[z][ind_loc[:, 0], ind_loc[:, 1]]
                for z in range(0, v)])
            local_box_add = local_box_add.flatten()
            local_box = np.hstack(
                (local_box, local_box_add)).astype('int32')

            label_grid = L_grid[v][y, x]
            local_pred = local_box[local_box < label_grid].astype('int32')

            P.append(local_pred)

        return P

    def __get_inv_ridge(self, P, X, alp):
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

    def get_assimilation_results(self, N_points, steps, std, size, atms_ens,
                                 atms_obs, Pr, atmospherical_models,
                                 nbrs=None):
        results = dict()
        for var in atms_ens.keys():
            R_ = sparse.diags(np.array(1/std[var]**2).repeat(N_points))
            by_var = list()
            for lvl, level in enumerate(PRESSURE_LEVELS_VALUES):
                Ms = [atmospherical_models[var][lvl, i] for i in range(4)]
                xb = atms_ens[var][lvl, 0].T.mean(1).reshape(-1, 1)
                rnd.seed(10)
                xs = list()
                points = np.arange(N_points)
                for k in range(steps):
                    if nbrs:
                        _nbrs = nbrs[var][lvl, k % 4]
                        idx = _nbrs.kneighbors(
                            X=xb.reshape(1, -1), return_distance=False)
                        Xb = atms_ens[var][lvl, k % 4, idx[0]].T
                    else:
                        Xb = atms_ens[var][lvl, k % 4].T
                    Obs_choosed = rnd.choice(points, size=size, replace=False)
                    Obs_choosed.sort()
                    DX = Xb-np.outer(xb, np.ones(Xb.shape[1]))
                    L, D_inv = self.__get_inv_ridge(Pr, DX, 0.1)
                    Bk_ = L.T.dot(D_inv.dot(L))
                    H = self.__get_H(Obs_choosed, N_points)
                    y = atms_obs[var][k, lvl].reshape(-1, 1)
                    d = H.dot(y - xb)
                    Rk_ = H.dot(R_.dot(H.T))
                    A = Bk_ + H.T.dot(Rk_.dot(H))
                    b = H.T.dot(Rk_.dot(d))
                    Z = splinalg.spsolve(A, b)
                    xa = xb + Z.reshape(-1, 1)
                    xs.append(xa)
                    xb = Ms[(k+1) % 4].predict(xa.reshape(1, -1))
                    xb = xb.reshape(-1, 1)
                print(f'Var: {var}. Lvl: {level}. Done!')
                by_var.append(xs)
            results[var] = np.array(by_var)
        return results
