import numpy as np
import pandas as pd
from datetime import datetime
import time

from hierarchSIR.training import log_posterior_probability
from hierarchSIR.utils import initialise_model, make_data_pySODM_compatible


strains = 2
seasons = ['2019-2020',]
season = seasons[0]
model_name = f'SIR-{strains}S'
immunity_linking = False
use_ED_visits = False

model = initialise_model(strains, immunity_linking=immunity_linking, season=season)
use_ED_visits = False

start_calibration = datetime(2019, 10, 1)
end_calibration = datetime(2020, 5, 1)

data, _, _, _ = make_data_pySODM_compatible(strains, use_ED_visits, start_calibration, end_calibration, season)
datasets = [data,]

par_names = ['rho_i', 'T_h', 'rho_h', 'f_R', 'f_I',  'beta', 'delta_beta_temporal']
par_bounds = [(1e-5,0.15), (0.5, 15), (1e-5,0.02), (0.01,0.99), (1e-9,1e-3), (0.01,1), (-1,1)]
par_hyperdistributions = ['lognorm', 'lognorm', 'lognorm', 'norm', 'lognorm', 'norm', 'norm']

lpp = log_posterior_probability(model, par_names, par_bounds, par_hyperdistributions, datasets, ['2019-2020',])

pars_model_0 = pd.read_csv('data/interim/calibration/single-season-optimal-parameters.csv', index_col=[0,1,2])
pars_0 = list(pars_model_0.loc[(model_name, immunity_linking, slice(None)), seasons].transpose().values.flatten().tolist())

# hyperparameters: use all seasons included as the default starting point
hyperpars_0 = pd.read_csv('data/interim/calibration/hyperparameters.csv', index_col=[0,1,2,3])
hyperpars_0 = hyperpars_0.loc[(model_name, immunity_linking, use_ED_visits, slice(None)), 'exclude_2024-2025'].values.tolist()

# combine
theta_0 = np.array(hyperpars_0 + pars_0)

assert np.isclose(lpp(theta_0), 35.6879, atol=1e-5), "lpp(theta_0) changed from 35.6879 to {lpp(theta_0)}"

# Time 1000 evaluations of lpp
start_time = time.time()
for _ in range(1000):
    lpp(theta_0)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time for 5000 evaluations: {elapsed_time:.4f} seconds")
print(f"Average time per evaluation: {elapsed_time/1000:.6f} seconds")
