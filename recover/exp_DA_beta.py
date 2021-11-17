#!/usr/bin/env python
# coding: utf-8



import os
import copy
import time
from pathlib import Path

import itertools as itr
import xarray as xr
import numpy as np
import numpy.random as rnd
import scipy.sparse as sparse

from sklearn import linear_model
from sklearn import neighbors
from scipy.sparse import linalg as splinalg

import warnings
warnings.simplefilter('ignore')


def get_data_structure(r, N_lat, N_lon, variables, L_grid):
    P = []
    for l in range(0, variables):
        # print(l)
        for y in range(0, N_lat):
            for x in range(0, N_lon):
                ran_lat = np.arange(max(0, y-r), min(N_lat, y+r+1))
                ran_lon = np.arange(x-r, x+r+1) % N_lon
                ind_loc = np.array([[j, i] for j in ran_lat for i in ran_lon])
                local_box = L_grid[l][ind_loc[:, 0], ind_loc[:, 1]]
                local_box_add = np.array([
                    L_grid[z][ind_loc[:, 0], ind_loc[:, 1]]
                    for z in range(0, l)])
                local_box_add = local_box_add.flatten()
                local_box = np.hstack(
                    (local_box, local_box_add)).astype('int32')

                label_grid = L_grid[l][y, x]
                local_pred = local_box[local_box < label_grid].astype('int32')

                P.append(local_pred)

    return P


def get_inv_ridge(P, X, alp):
    n = len(P)
    I = list(range(n))
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

        I.extend(N_pred*list([i]))
        J.extend(pred)
        LV.extend(-lr.coef_.flatten())

        DV[i] = 1/np.var(res)

    I = np.array(I).astype('int32')
    J = np.array(J).astype('int32')

    L = sparse.coo_matrix((LV, (I, J)))
    D_inv = sparse.diags(DV)

    return L, D_inv


def get_train_test(path, slice_train, slice_test, pressure_lvls):
    FILES = os.listdir(path)
    list.sort(FILES)
    atmospherical_variables = dict()
    atmospherical_variables_test = dict()
    for file in FILES:
        variable = file.split('.')[0]
        variable_DS = xr.open_dataset(path/f'{file}')[variable]
        atmospherical_variables[variable] = variable_DS.sel(
            level=pressure_lvls, time=slice_train).values
        atmospherical_variables_test[variable] = variable_DS.sel(
            level=pressure_lvls, time=slice_test).values
    return atmospherical_variables, atmospherical_variables_test


def group_by_hour(atmos_dict):
    key = list(atmos_dict.keys())[0]
    steps, nlvl, nlat, nlon = atmos_dict[key].shape
    days = steps//4
    atmospherical_H = dict()
    for key, values in atmos_dict.items():
        atmospherical = list()
        for k in range(nlvl):
            var = np.reshape(values[:, k, :, :], [steps, nlon*nlat]).T
            atmospherical.append([[var[:, d+i] for d in range(days-1)]
                                  for i in range(4)])
            atmospherical[-1].append(var[:, 1:days].T)
        atmospherical_H[key] = np.array(atmospherical)

    return atmospherical_H


def get_fitted_models(regressor, atmos_H, nlvl):
    atmos_models = dict()
    for key in atmos_H.keys():
        models = list()
        for k in range(nlvl):
            models.append(list())
            for i in range(4):
                rg = copy.copy(regressor)
                X = atmos_H[key][k, i]
                y = atmos_H[key][k, i+1]
                models[-1].append(rg.fit(X=X, y=y))
        atmos_models[key] = np.array(models)
    return atmos_models


def get_nbrs_models(nbrs, atmos_H, nlvl):
    _nbrs_models = dict()
    for key in atmos_H.keys():
        models = list()
        for k in range(nlvl):
            models.append(list())
            for i in range(4):
                _nbrs = copy.copy(nbrs)
                X = atmos_H[key][k, i]
                models[-1].append(_nbrs.fit(X=X))
        _nbrs_models[key] = np.array(models)
    return _nbrs_models


def get_H(idx, n_points):
    H = sparse.lil_matrix((idx.size, n_points))
    for i, j in enumerate(idx):
        H[i, j] = 1
    return H.tocsc()


def get_assimilation_results(N_points, steps, std, size, atms_ens, atms_obs,
                             Pr, atmospherical_models, nbrs=None):
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
                L, D_inv = get_inv_ridge(Pr, DX, 0.1)
                Bk_ = L.T.dot(D_inv.dot(L))
                H = get_H(Obs_choosed, N_points)
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


# PRESSURE_LEVELS_VALUES = [925,850,700,500,300,200,100]
PRESSURE_LEVELS_VALUES = [925]
N_LVL = len(PRESSURE_LEVELS_VALUES)
N_LAT = 73
N_LON = 144
N_STEPS = 15

N_POINTS = N_LAT*N_LON

TIME_TRAIN = slice('2020-01-01', '2020-08-23')
TIME_TEST = slice('2020-08-24', '2020-09-06')
L_grid = [np.arange(N_POINTS).reshape(N_LAT, N_LON)]
PATH = Path.cwd().parent
std = dict({
    'air': 1,
    'uwnd': 1,
    'vwnd': 1,
    'rhum': 0.0001,
})

atmos = np.load(PATH/'data/NOAA/Atmospherical_Conditions/atmos.npz')
atmos_test = np.load(PATH/'data/NOAA/Atmospherical_Conditions/atmos_test.npz')
atmos = group_by_hour(atmos)
atmos_test = dict({key: value.reshape((-1, N_LVL, N_LAT*N_LON))
                   for key, value in atmos_test.items()})


Ns = [50, 100, 150]
Wts = ['distance', 'uniform']
obs = [5, 10, 15]
rs = [3, 5, 7]

for wt, n, r, p in itr.product(Wts, Ns, rs, obs):
    SIZE = N_POINTS*p//100
    Pr = get_data_structure(r, N_LAT, N_LON, 1, L_grid)
    nbrs = neighbors.NearestNeighbors(n_neighbors=n)
    nbrs_models = get_nbrs_models(nbrs, atmos, N_LVL)
    rg = neighbors.KNeighborsRegressor(
        n_neighbors=n, weights=wt, algorithm='brute')
    atmos_models = get_fitted_models(rg, atmos, N_LVL)
    results = get_assimilation_results(
        N_POINTS, N_STEPS, std, SIZE, atmos, atmos_test, Pr, atmos_models,
        nbrs_models)
    np.savez(
        Path.cwd().parent/f'results/results_analysis_{wt}_n{n}_r{r}_obs{p}',
        **results)
    print(f"Done for {wt} with n={n}, r={r} and {p}% obs")
