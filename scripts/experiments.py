import sys
import pickle
import numpy as np
from Markovian import Markovian
from pathlib import Path
import sklearn.utils as skutils

PATH = Path(__file__).parent.resolve()

rg, r, p_obs = sys.argv[1:]
r, p_obs = int(r), float(p_obs)

# regressors = dict({
#     name: _class for name, _class in
#     skutils.all_estimators(type_filter="regressor")
# }

# with open(PATH/'../regressors.pkl', 'wb') as handle:
#     pickle.dump(regressors, handle)

with open(PATH/'../regressors.pkl', 'rb') as handle:
    REGRESSORS = pickle.load(handle)

# PRESSURE_LEVELS_VALUES = [925, 850, 700, 500, 300, 200, 100]
PRESSURE_LEVELS_VALUES = [925]
N_LAT = 73
N_LON = 144

VAR = ['air', 'uwnd', 'vwnd', 'rhum']
VAR_STD = dict({
    'air': 1,
    'uwnd': 1,
    'vwnd': 1,
    'rhum': 0.0001,
})

VAR_DATA_TRAIN = np.load(PATH/'../data/atmos_cond_train.npz')
VAR_DATA_TEST = np.load(PATH/'../data/atmos_cond_test.npz')
STEPS, _ = VAR_DATA_TEST[list(VAR_DATA_TEST.keys())[0]].shape

var_models = dict({
    var: Markovian(
        N_lat=N_LAT, N_lon=N_LON, regressor=REGRESSORS[rg]).fit(vals)
    for var, vals in VAR_DATA_TRAIN.items()})

da_results = dict({
    var: model.EnKFMC_results(
        steps=STEPS,
        std=VAR_STD[var],
        p_obs=p_obs,
        r=r,
        atms_ens=VAR_DATA_TRAIN[var],
        atms_obs=VAR_DATA_TEST[var],
        seed=10)
    for var, model in var_models.items()
})

np.savez(
    PATH/f'../results/results_analysis_{rg}_r{r}_obs{p_obs}', **da_results)

print(f"Done for {rg} with r={r} and p_obs={p_obs}")
