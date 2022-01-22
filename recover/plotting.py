#!/usr/bin/env python
# coding: utf-8
import time
import matplotlib
import numpy as np
import xarray as xr
import seaborn as sns
import itertools as itr
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from pathlib import Path

matplotlib.rcParams["mathtext.fontset"] = 'cm'
matplotlib.rcParams.update({'font.size': 14})
sns.set_style("whitegrid")

CWD = Path.cwd()


def error(y_true_, y_pred_):
    # u = ((y_true - y_pred) ** 2).sum(axis = 0)
    # v = ((y_true - y_true.mean(axis = 0)) ** 2).sum(axis = 0)
    # return 1-u/v
    return (np.abs(y_true_-y_pred_)/np.maximum(
        np.abs(y_true_), 1)).mean(axis=0)


def format_time(x):
    formatted = time.strftime(
        "%m-%d %H:%M:%S",
        time.strptime(np.datetime_as_string(x), "%Y-%m-%dT%H"))
    return formatted


# # Loading Data
PRESSURE_LEVELS_VALUES = [925, 850, 700, 500, 300, 200, 100]
N_LVLS = len(PRESSURE_LEVELS_VALUES)
VARS = ["air", "rhum", "uwnd", "vwnd"]
N_LAT = 73
N_LON = 144
N_POINTS = N_LAT*N_LON
PATH = CWD.parent/'data/NOAA/Atmospherical_Conditions/'

print("Reading files")
atmospherical_variables_test = dict({
    var: xr.open_dataset(
        PATH/f"{var}.2020.nc")[var].sel(
        level=PRESSURE_LEVELS_VALUES).values for var in VARS
})

N_STEPS = atmospherical_variables_test[VARS[0]].shape[0]

VAR_NAME = {
    'air': '$T$',
    'uwnd': '$u$',
    'vwnd': '$v$',
    'rhum': '$\\phi$'
    }

Ks = range(10, 51, 10)
rs = range(1, 6, 2)
Ls = [5]

print("Plotting...")
for var, values in atmospherical_variables_test.items():
    for L in Ls:
        plt.figure(figsize=(12, 8))
        for K, r in itr.product(Ks, rs):
            test = np.load(
                CWD.parent/f"results/exp_Markovian_zeta/results_L{L}_K{K}_r{r}.npz")
            y_true = values[:, 0].reshape((N_STEPS, N_POINTS))
            y_pred = test[var][0]
            idx = min(y_true.shape[0], y_pred.shape[0])
            y_true = y_true[:idx]
            y_pred = y_pred[:idx]
            errors = (
                (y_pred-y_true)**2).mean(axis=1)/linalg.norm(y_true, axis=1)
            plt.plot(np.log10(errors), label=f"({K}, {r})")
        plt.legend(title='($K$, $r$)')
        plt.ylabel('$\\log_{10}(\\epsilon_{k})$', fontdict={'fontsize': 20})
        plt.ylim(-4, 0)
        plt.title(VAR_NAME[var], fontdict={"fontsize": 30})
        plt.savefig(CWD.parent/f'figures/mse_L{L}_{var}', dpi=250)
        print(f"All done for {var} with L={L}")
