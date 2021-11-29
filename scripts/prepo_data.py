import os
import xarray as xr
import numpy as np
from pathlib import Path
import sklearn.utils as skutils

PATH = Path(__file__).parent.resolve()
VARS = ['air', 'rhum', 'uwnd', 'vwnd']
NLAT = 73
NLON = 144

VAR_DATA_TRAIN = {var: xr.concat([xr.open_dataset(
    PATH.parent/f'data/{file}')[var].sel(
        level=925, time=slice('1996-01-01', '2020-12-31'))
        for file in os.listdir(PATH.parent/'data')
        if file.split('.')[0] == var],
        dim='time').values.reshape((-1, NLAT*NLON)) for var in VARS}
np.savez(PATH.parent/'data/atmos_cond_train.npz', **VAR_DATA_TRAIN)

VAR_DATA_TEST = {var: xr.concat([xr.open_dataset(
    PATH.parent/f'data/{file}')[var].sel(
        level=925, time=slice('2021-01-01', '2021-12-31'))
        for file in os.listdir(PATH.parent/'data')
        if file.split('.')[0] == var],
        dim='time').values.reshape((-1, NLAT*NLON)) for var in VARS}
np.savez(PATH.parent/'data/atmos_cond_test.npz', **VAR_DATA_TEST)


