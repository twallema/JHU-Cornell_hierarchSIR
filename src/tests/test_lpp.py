import time
import numpy as np
import pandas as pd
from datetime import datetime
import hierarchSIR.training as optim
from hierarchSIR.utils import initialise_model, make_data_pySODM_compatible

# Define model settings
strains = 2
seasons = ['2017-2018', '2018-2019']
season = seasons[0]
model_name = f'SIR-{strains}S'
immunity_linking = False
use_ED_visits = False

# initialise model
model = initialise_model(strains, immunity_linking=immunity_linking, season=season)

# get and format data
start_calibrations = [datetime(int(season[0:4]), 10, 1) for season in seasons]
end_calibrations = [datetime(int(season[0:4])+1, 5, 1) for season in seasons]
datasets = []
for start_calibration, end_calibration, season in zip(start_calibrations, end_calibrations, seasons):
    data, _, _, _ = make_data_pySODM_compatible(strains, use_ED_visits, start_calibration, end_calibration, season)
    datasets.append(data)

# define parameters, their bounds and their hyperdistributions
par_names = ['rho_i', 'T_h', 'rho_h', 'f_R', 'f_I',  'beta', 'delta_beta_temporal']
par_bounds = [(1e-5,0.15), (0.5, 15), (1e-5,0.02), (0.01,0.99), (1e-9,1e-3), (0.01,1), (-1,1)]
par_hyperdistributions = ['beta', 'gamma', 'lognorm', 'norm', 'lognorm', 'norm', 'norm']

# set up lpp function
lpp = optim.log_posterior_probability(model, par_names, par_bounds, par_hyperdistributions, datasets, seasons)

# get initial guess parameters
pars_model_0 = pd.read_csv('../../data/interim/calibration/single-season-optimal-parameters.csv', index_col=[0,1,2])
pars_0 = list(pars_model_0.loc[(model_name, immunity_linking, slice(None)), seasons].transpose().values.flatten().tolist())

# get starting point hyperparameters
hyperpars_0 = pd.read_csv('../../data/interim/calibration/hyperparameters.csv', index_col=[0,1,2,3])
hyperpars_0 = hyperpars_0.loc[(model_name, immunity_linking, use_ED_visits, slice(None)), 'exclude_None'].values.tolist()

# combine 
theta_0 = np.array(hyperpars_0 + pars_0)

# run lpp function
lpp(theta_0)

print(lpp(theta_0))

# TODO: run some more assertions and tests