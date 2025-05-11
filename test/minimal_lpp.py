import numpy as np
import pandas as pd
from datetime import datetime
import time

import hierarchSIR.training_optim as optim
import hierarchSIR.training as orig

from hierarchSIR.utils import initialise_model, make_data_pySODM_compatible


strains = 2
seasons = ['2017-2018', '2018-2019']
season = seasons[0]
model_name = f'SIR-{strains}S'
immunity_linking = False
use_ED_visits = False

model = initialise_model(strains, immunity_linking=immunity_linking, season=season)
use_ED_visits = False

start_calibrations = [datetime(int(season[0:4]), 10, 1) for season in seasons]
end_calibrations = [datetime(int(season[0:4])+1, 5, 1) for season in seasons]
datasets = []
for start_calibration, end_calibration, season in zip(start_calibrations, end_calibrations, seasons):
    data, _, _, _ = make_data_pySODM_compatible(strains, use_ED_visits, start_calibration, end_calibration, season)
    datasets.append(data)

par_names = ['rho_i', 'T_h', 'rho_h', 'f_R', 'f_I',  'beta', 'delta_beta_temporal']
par_bounds = [(1e-5,0.15), (0.5, 15), (1e-5,0.02), (0.01,0.99), (1e-9,1e-3), (0.01,1), (-1,1)]
par_hyperdistributions = ['beta', 'gamma', 'lognorm', 'norm', 'lognorm', 'norm', 'norm']

lpp_orig = orig.log_posterior_probability(model, par_names, par_bounds, par_hyperdistributions, datasets, seasons)
lpp_optim = optim.log_posterior_probability(model, par_names, par_bounds, par_hyperdistributions, datasets, seasons)

pars_model_0 = pd.read_csv('../data/interim/calibration/single-season-optimal-parameters.csv', index_col=[0,1,2])
pars_0 = list(pars_model_0.loc[(model_name, immunity_linking, slice(None)), seasons].transpose().values.flatten().tolist())

# hyperparameters: use all seasons included as the default starting point
hyperpars_0 = pd.read_csv('../data/interim/calibration/hyperparameters.csv', index_col=[0,1,2,3])
hyperpars_0 = hyperpars_0.loc[(model_name, immunity_linking, use_ED_visits, slice(None)), 'exclude_2024-2025'].values.tolist()

# combine
theta_0 = np.array(hyperpars_0 + pars_0)

assert np.isclose(lpp_orig(theta_0), lpp_optim(theta_0), atol=1e-5), f"Results differ: {lpp_orig(theta_0)} vs {lpp_optim(theta_0)}"

# Time 25x200 evaluations of lpp
def time_function(func, inner, outer):
    times = []
    for _ in range(outer):
        start_time = time.time()
        for _ in range(inner):
            func(theta_0)
        end_time = time.time()
        times.append(end_time - start_time)
    return times

def print_time_statistics(times):
    print(f"\tMean time: {np.mean(times):.4f} seconds")
    print(f"\tMedian time: {np.median(times):.4f} seconds")
    print(f"\tStandard deviation: {np.std(times):.4f} seconds")
    print(f"\tMinimum time: {np.min(times):.4f} seconds")
    print(f"\tMaximum time: {np.max(times):.4f} seconds")

print("\nTiming results for original function:")
times_orig = time_function(lpp_orig, 25, 200)
print_time_statistics(times_orig)

print("\nTiming results for optimized function:")
times_optim = time_function(lpp_optim, 25, 200)
print_time_statistics(times_optim)

print("Relative improvement:")
print(f"\tMean: {(np.mean(times_optim) - np.mean(times_orig)) / np.mean(times_orig):.4f}")
