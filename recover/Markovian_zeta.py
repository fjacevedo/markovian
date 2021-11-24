#!/usr/bin/env python
# coding: utf-8

import warnings
import numpy as np
import xarray as xr
import itertools as itr
import sklearn.base as base
import sklearn.neighbors as neighbors
from pathlib import Path

warnings.simplefilter('ignore')
CWD = Path.cwd()


def gaussian_wts(dists, r=1, alpha=0.01):
    return np.maximum(np.exp(-0.5*(dists/r)**2), alpha)


def group_by_hour(atmos_dict, l_=4):
    key = list(atmos_dict.keys())[0]
    steps, nlvl, nlat, nlon = atmos_dict[key].shape
    n_pts = nlat*nlon
    days = steps//l_
    _atmospherical_H = dict()
    for key, values in atmos_dict.items():
        atmospherical = np.zeros((nlvl, l_+1, days-1, n_pts))
        for k in range(nlvl):
            var = np.reshape(values[:, k, :, :], (steps, n_pts)).T
            atmospherical[k, :-1, :, :] = np.array(
                [[var[:, d+i] for d in range(days-1)] for i in range(l_)])
            atmospherical[k, -1, :, :] = var[:, 1:days].T
        _atmospherical_H[key] = atmospherical.copy()

    return _atmospherical_H


def fit_predict(regressor, train_dict, test_dict):
    key = list(train_dict.keys())[0]
    nlvl, l_, _, n_pts = train_dict[key].shape
    l_ -= 1
    steps = l_*(test_dict[key].shape[2] + 1)
    _pred_dict = dict()
    for key, train in train_dict.items():
        pred = np.zeros((nlvl, steps, n_pts))
        for k in range(nlvl):
            models = [
                base.clone(regressor).fit(
                    X=train[k, i],
                    y=train[k, i+1]) for i in range(l_)]
            X_test = test_dict[key][k, 0, 0].reshape(1, -1)
            for i in range(steps):
                X_test = models[i % l_].predict(X_test)
                pred[k, i] = X_test.copy()
        _pred_dict[key] = pred.copy()
    return _pred_dict


# # Loading Data
PRESSURE_LEVELS_VALUES = [925, 850, 700, 500, 300, 200, 100]
N_LVLS = len(PRESSURE_LEVELS_VALUES)
VARS = ["air", "rhum", "uwnd", "vwnd"]
N_LAT = 73
N_LON = 144
N_POINTS = N_LAT*N_LON
TIME_SLICE = slice('2015-01-01', '2019-12-31')
TIME_TEST = slice('2020-01-01', '2020-12-31')
PATH = CWD.parent/'data/NOAA/Atmospherical_Conditions/'

atmospherical_variables = dict()
atmospherical_variables_test = dict()

print("Reading files")
for var in VARS:
    variable_DS = xr.concat([
        xr.open_dataset(PATH/f'{file}') for file in PATH.glob(f"{var}.*.nc")
        ], dim="time")[var]
    atmospherical_variables[var] = variable_DS.sel(
        level=PRESSURE_LEVELS_VALUES, time=TIME_SLICE).values
    atmospherical_variables_test[var] = variable_DS.sel(
        level=PRESSURE_LEVELS_VALUES, time=TIME_TEST).values

Ls = range(3, 15)
Ks = range(10, 51, 10)
rs = range(1, 6, 2)

for L in Ls:
    if L < 6:
        continue

    print(f"Grouping by hour - L={L}")
    atmospherical_H = group_by_hour(atmospherical_variables, L)
    atmospherical_H_test = group_by_hour(atmospherical_variables_test, L)

    for K, r in itr.product(Ks, rs):
        # # Fitting models variables x level x hour
        # models trained with data from 2015-01-01 to 2019-12-31
        if L == 6 and K < 30:
            continue

        print(f"Training Models - K={K} and r={r}")
        rg = neighbors.KNeighborsRegressor(
            n_neighbors=K,
            weights=lambda dists: gaussian_wts(dists, r=r),
            n_jobs=-1)

        pred_dict = fit_predict(
            rg, atmospherical_H, atmospherical_H_test)

        print("> Done!")

        f = f"/work/syseng/users/fjacevedo/results_Markovian_zeta/results_L{L}_K{K}_r{r}.npz"
        np.savez(f, **pred_dict)
