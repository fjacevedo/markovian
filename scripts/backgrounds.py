# !/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
from pathlib import Path
from Markovian import Markovian
from datetime import datetime as dt

PATH = Path(__file__).parent.resolve()
REGRESSORS = ["Ridge", "RidgeCV", "LinearRegression",
              "GaussianProcessRegressor", "KNeighborsRegressor"]
VAR_DATA_TRAIN = np.load(PATH.parent/'data/atmos_cond_train.npz')

N_LAT = 73
N_LON = 144
STEPS = 124
VAR = ['air', 'uwnd', 'vwnd', 'rhum']

with open(PATH.parent/'regressors.pkl', 'rb') as handle:
    ALL_REGRESSORS = pickle.load(handle)


for rg in REGRESSORS:
    print(
        '{}: Initiates models {}'.format(
            dt.now().strftime("%d/%m/%Y %H:%M:%S"), rg))
    back_results = dict()
    if 'n_jobs' in ALL_REGRESSORS[rg]().get_params():
        var_models = dict({
            var: Markovian(
                N_lat=N_LAT,
                N_lon=N_LON,
                regressor=ALL_REGRESSORS[rg],
                n_jobs=-1).fit(vals)
            for var, vals in VAR_DATA_TRAIN.items()
        })
    else:
        var_models = dict({
            var: Markovian(
                N_lat=N_LAT, N_lon=N_LON, regressor=ALL_REGRESSORS[rg]).fit(vals)
            for var, vals in VAR_DATA_TRAIN.items()
        })
    for var in VAR:
        result = np.zeros((STEPS, N_LAT*N_LON))
        xb = VAR_DATA_TRAIN[var].mean(axis=0).reshape(1, -1)
        # print(xb.shape)
        for step in range(STEPS):
            xb = var_models[var].predict(xb, current=step % 4)
            result[step] = xb.ravel()
        back_results[var] = result
        print('{}: Done for {}'.format(
            dt.now().strftime("%d/%m/%Y %H:%M:%S"), var))
    np.savez(PATH.parent/f"data/back/{rg}.npz", **back_results)
