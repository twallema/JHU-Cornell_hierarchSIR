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
from hierarchSIR.utils import initialise_model, make_data_pySODM_compatible

##############
## Settings ##
##############

# calibration settings
strains = True
use_ED_visits = True                                                                                    # use both ED admission (hospitalisation) and ED visits (ILI) data 
seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2023-2024']    # season to include in calibration excercise
start_calibration_month = 10                                                                             # start calibration on month 10, day 1
end_calibration_month = 5                                                                                # end calibration on month 5, day 1

# Define number of chains
max_n = 40000
n_chains = 400
pert = 0.10
run_date = datetime.today().strftime("%Y-%m-%d")
identifier = 'exclude-2024-2025'
print_n =  40000
backend =  None
discard = 36000
thin = 100
processes = int(os.environ.get('NUM_CORES', '16'))

# Make folder structure
if strains:
    model_name = 'SIR-2S'
else:
    model_name = 'SIR-1S'

if use_ED_visits:
    samples_path=fig_path=f'../data/interim/calibration/hierarchical-training/{model_name}/use_ED_visits/' # Path to backend
else:
    samples_path=fig_path=f'../data/interim/calibration/hierarchical-training/{model_name}/not_use_ED_visits/' # Path to backend
# check if samples folder exists, if not, make it
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

################
## Setup data ##
################

# convert to a list of start and enddates (datetime)
start_calibrations = [datetime(int(season[0:4]), start_calibration_month, 1) for season in seasons]
end_calibrations = [datetime(int(season[0:4])+1, end_calibration_month, 1) for season in seasons]

# get data
datasets = []
for start_calibration, end_calibration, season in zip(start_calibrations, end_calibrations, seasons):
    data, _, _, _ = make_data_pySODM_compatible(strains, use_ED_visits, start_calibration, end_calibration, season)
    datasets.append(data)

#################
## Setup model ##
#################

model = initialise_model(strains=strains)

##########################################
## Setup posterior probability function ##
##########################################

# define model parameters to calibrate to every season and their bounds
# not how we're not cutting out the parameters associated with the ED visit data
par_names = ['rho_i', 'T_h', 'rho_h', 'f_R', 'f_I', 'beta', 'delta_beta_temporal']
par_bounds = [(1e-5,0.15), (0.1, 15), (1e-5,0.02), (0.001,0.999), (1e-9,1e-3), (0.01,1), (-1,1)]
par_hyperdistributions = ['gamma', 'expon', 'gamma', 'beta', 'gamma', 'normal', 'normal']

# setup lpp function
lpp = log_posterior_probability(model, par_names, par_bounds, par_hyperdistributions, datasets)

####################################
## Fetch initial guess parameters ##
####################################

# parameters: get optimal independent fit with informative prior on R0
pars_model_0 = pd.read_csv('../data/interim/calibration/single-season-optimal-parameters.csv', index_col=1)
pars_model_0 = pars_model_0[pars_model_0['strains']==strains][seasons]
pars_0 = list(pars_model_0.transpose().values.flatten())

# hyperparameters: use all seasons included as starting point
hyperpars_0 = pd.read_csv('../data/interim/calibration/hyperparameters.csv')
hyperpars_0 = list(hyperpars_0.loc[((hyperpars_0['model'] == model_name) & (hyperpars_0['use_ED_visits'] == use_ED_visits)), (['parameter', 'exclude-2024-2025'])].set_index('parameter').squeeze())

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
                # write median hyperpars to .csv
                hyperpars_names = []
                hyperpars_values = []
                for hyperpar_name, hyperpar_shape in lpp.hyperpar_shapes.items():
                    # append value
                    hyperpars_values.append(samples.median(dim=['chain', 'iteration'])[hyperpar_name].values.tolist())
                    # append name
                    hyperpars_names.extend([f'{hyperpar_name}_{i}' if hyperpar_shape[0] > 1 else f'{hyperpar_name}' for i in range(hyperpar_shape[0])])
                hyperpars_values = np.hstack(hyperpars_values)
                # save to .csv
                pd.Series(index=hyperpars_names, data=hyperpars_values, name=identifier).to_csv(samples_path+str(identifier)+'_HYPERDIST_'+run_date+'.csv')
                # .. visualise hyperdistributions
                hyperdistributions(samples, samples_path+str(identifier)+'_HYPERDIST_'+run_date+'.pdf', lpp.par_shapes, lpp.hyperpar_shapes, par_hyperdistributions, par_bounds, 100)
                # ..generate traceplots
                traceplot(samples, lpp.par_shapes, lpp.hyperpar_shapes, samples_path, identifier, run_date)
                # ..generate goodness-of-fit
                plot_fit(model, datasets, lpp.simtimes, samples, model.parameter_shapes, samples_path, identifier, run_date,
                         lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data, lpp.corresponding_model_states)