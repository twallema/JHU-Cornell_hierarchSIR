"""
This script does..
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import emcee
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import get_context
from hierarchSIR.training import log_posterior_probability, dump_sampler_to_xarray, traceplot, plot_fit, hyperdistributions
from hierarchSIR.utils import initialise_model, get_NC_influenza_data

##############
## Settings ##
##############

# calibration settings
strains = False
use_ED_visits = True                                                                                    # use both ED admission (hospitalisation) and ED visits (ILI) data 
seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2023-2024']    # season to include in calibration excercise
start_calibration_month = 10                                                                             # start calibration on month 10, day 1
end_calibration_month = 5                                                                                # end calibration on month 5, day 1

# Define number of chains
max_n = 25000
n_chains = 400
pert = 0.01
run_date = datetime.today().strftime("%Y-%m-%d")
identifier = 'exclude-2024-2025'
print_n =  1000
backend =  None
discard = 0
thin = 1
processes = int(os.environ.get('NUM_CORES', '16'))

# Make folder structure
if use_ED_visits:
    samples_path=fig_path=f'../data/interim/calibration/hierarchical-training/oneStrain/use_ED_visits/' # Path to backend
else:
    samples_path=fig_path=f'../data/interim/calibration/hierarchical-training/oneStrain/not_use_ED_visits/' # Path to backend
# check if samples folder exists, if not, make it
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

################
## Setup data ##
################

# convert to a list of start and enddates (datetime)
start_calibrations = [datetime(int(season[0:4]), start_calibration_month, 1) for season in seasons]
end_calibrations = [datetime(int(season[0:4])+1, end_calibration_month, 1) for season in seasons]

# gather datasets per season in a list
datasets = [get_NC_influenza_data(start_calibration, end_calibration, season) for start_calibration, end_calibration, season in zip(start_calibrations, end_calibrations, seasons)]

datasets = []
for start_calibration, end_calibration, season in zip(start_calibrations, end_calibrations, seasons):
    # attach I_inc and H_inc
    if strains:
        # pySODM formatting for flu A
        flu_A = get_NC_influenza_data(start_calibration, end_calibration, season)['H_inc_A']
        flu_A = flu_A.rename('H_inc') # pd.Series needs to have matching model state's name
        flu_A = flu_A.reset_index()
        flu_A['strain'] = 0
        flu_A = flu_A.set_index(['date', 'strain']).squeeze()
        # pySODM formatting for flu B
        flu_B = get_NC_influenza_data(start_calibration, end_calibration, season)['H_inc_B']
        flu_B = flu_B.rename('H_inc') # pd.Series needs to have matching model state's name
        flu_B = flu_B.reset_index()
        flu_B['strain'] = 1
        flu_B = flu_B.set_index(['date', 'strain']).squeeze()
        # attach all datasets
        datasets.append([get_NC_influenza_data(start_calibration, end_calibration, season)['I_inc'], flu_A, flu_B])
    else:
        datasets.append([get_NC_influenza_data(start_calibration, end_calibration, season)['I_inc'], get_NC_influenza_data(start_calibration, end_calibration, season)['H_inc']])
    # omit I_inc
    if not use_ED_visits:
        datasets[-1] = datasets[-1][1:]

#################
## Setup model ##
#################

model = initialise_model(strains=strains)

##########################################
## Setup posterior probability function ##
##########################################

# define model parameters to calibrate to every season and their bounds
# not how we're not cutting out the parameters associated with the ED visit data
par_names = ['rho_i', 'T_h', 'rho_h', 'beta', 'f_R', 'f_I', 'delta_beta_temporal']
par_bounds = [(1e-5,0.15), (0.1, 15), (1e-5,0.02), (0.01,1), (0.001,0.999), (1e-9,1e-3), (-1,1)]
par_hyperdistributions = ['gamma', 'expon', 'gamma', 'normal', 'beta', 'gamma', 'normal']

# setup lpp function
lpp = log_posterior_probability(model, par_names, par_bounds, par_hyperdistributions, datasets)

####################################
## Fetch initial guess parameters ##
####################################

# get independent fit parameters
pars_model_0 = pd.read_csv('../data/interim/calibration/single-season-optimal-parameters-oneStrain.csv', index_col=0)[seasons]

# manually tweak beta (this model has no age groups)
pars_model_0.loc['beta'] = 21 * pars_model_0.loc['beta']

# parameters
pars_0 = list(pars_model_0.transpose().values.flatten())

# hyperparameters
hyperpars_0 = [
               5.0, 3.0e-02,                                                                # rho_i
               1.7,                                                                         # T_h
               5.7, 3.0e-03,                                                                # rho_h
               0.55, 0.10,                                                                  # beta
               12.0, 16.5,                                                                  # f_R
               4.3, 2.8e-05,                                                                # f_I
               -0.06, -0.04, -0.02, 0.01, 0.13, -0.13, 0.02, 0.11, 0.03, 0.03, 0.08, -0.04, # delta_beta_temporal_mu
               0.04, 0.05, 0.03, 0.05, 0.1, 0.13, 0.11, 0.10, 0.14, 0.10, 0.23, 0.10,       # delta_beta_temporal_sigma
                ]

# combine
theta_0 = hyperpars_0 + pars_0

###################
## Setup sampler ##
###################

# Generate random perturbations from a normal distribution
perturbations = np.random.normal(
        loc=1, scale=pert, size=(n_chains, len(theta_0))
    )

# Apply perturbations to create the 2D array
pos = np.array(theta_0)[None, :] * perturbations
nwalkers, ndim = pos.shape

# By default: set up a fresh hdf5 backend in samples_path
if not backend:
    fn_backend = str(identifier)+'_BACKEND_'+run_date+'.hdf5'
    backend = emcee.backends.HDFBackend(samples_path+fn_backend)
# If user provides an existing backend: continue sampling 
else:
    try:
        backend = emcee.backends.HDFBackend(samples_path+backend)
        pos = backend.get_chain(discard=discard, thin=thin, flat=False)[-1, ...]
    except:
        raise FileNotFoundError("backend not found.")    

# setup sampler
if __name__ == '__main__':
    with get_context("spawn").Pool(processes=processes) as pool:
        # setup sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lpp, backend=backend, pool=pool,
                                        moves=[(emcee.moves.DEMove(), 0.5*0.9),(emcee.moves.DEMove(gamma0=1.0), 0.5*0.1),
                                               (emcee.moves.StretchMove(live_dangerously=True), 0.50)]
                                        )
        # sample
        for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True, skip_initial_state_check=True):

            if sampler.iteration % print_n:
                continue
            else:
                # every print_n steps do..
                # ..dump samples
                samples = dump_sampler_to_xarray(sampler.get_chain(discard=discard, thin=thin), samples_path+str(identifier)+'_SAMPLES_'+run_date+'.nc', lpp.hyperpar_shapes, lpp.par_shapes, seasons)
                # .. visualise hyperdistributions
                hyperdistributions(samples, samples_path+str(identifier)+'_HYPERDIST_'+run_date+'.pdf', lpp.par_shapes, par_hyperdistributions, par_bounds, 100)
                # ..generate traceplots
                traceplot(samples, lpp.par_shapes, lpp.hyperpar_shapes, samples_path, identifier, run_date)
                # ..generate goodness-of-fit
                plot_fit(model, datasets, lpp.simtimes, samples, par_names, samples_path, identifier, run_date,
                         lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data, lpp.corresponding_model_states)