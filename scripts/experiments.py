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

with open(PATH.parent/'regressors.pkl', 'rb') as handle:
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

VAR_DATA_TRAIN = np.load(PATH.parent/'data/atmos_cond_train.npz')
VAR_DATA_TEST = np.load(PATH.parent/'data/atmos_cond_test.npz')
STEPS, _ = VAR_DATA_TEST[list(VAR_DATA_TEST.keys())[0]].shape

print('{} with r={} and p_obs={}: Training models'.format(rg, r, int(p_obs*100)))
var_models = dict({
    var: Markovian(
        N_lat=N_LAT, N_lon=N_LON, regressor=REGRESSORS[rg]).fit(vals)
    for var, vals in VAR_DATA_TRAIN.items()})

print('{} with r={} and p_obs={}: Starting DA process'.format(rg, r, int(p_obs*100)))
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

print(print('{} with r={} and p_obs={}: Saving results'.format(rg, r, int(p_obs*100))))
np.savez(PATH.parent/'results/{}_{}_{}'.format(rg, r, int(p_obs*100)),
         **da_results)

print(print('{} with r={} and p_obs={}: Done!'.format(rg, r, int(p_obs*100))))
