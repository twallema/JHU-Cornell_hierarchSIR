"""
This script calibrates the influenza model to North Carolina ED admission and ED visits data
It automatically calibrates to incrementally larger datasets between `start_calibration` and `end_calibration`
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import sys,os
import random
import emcee
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime as datetime
# pySODM functions
from pySODM.optimization import nelder_mead
from pySODM.optimization.utils import assign_theta, add_poisson_noise
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_normal, log_prior_uniform, log_prior_gamma, log_prior_normal, log_prior_beta
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler
# hierarchSIR functions
from hierarchSIR.utils import initialise_model, pySODM_to_hubverse, plot_fit, make_data_pySODM_compatible # influenza model

#####################
## Parse arguments ##
#####################

import argparse
# helper function
def str_to_bool(value):
    """Convert string arguments to boolean (for SLURM environment variables)."""
    return value.lower() in ["true", "1", "yes"]

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--strains", type=str_to_bool, help="Use a two strain model.")
parser.add_argument("--immunity_linking", type=str_to_bool, help="Use an immunity linking function.")
parser.add_argument("--use_ED_visits", type=str_to_bool, help="Use ED visit data (ILI) in addition to ED admission data (hosp. adm.).")
parser.add_argument("--informed", type=str_to_bool, default=None, help="Use priors informed by posterior hyperdistributions.")
parser.add_argument("--hyperparameters", type=str, default=None, help="Name of posterior hyperdistribution. Provide a valid column name in 'summary-hyperparameters.csv' to load the hyperdistributions.")
parser.add_argument("--season", type=str, help="Season to calibrate to. Format: '20XX-20XX'")
args = parser.parse_args()

# assign to desired variables
strains = args.strains
immunity_linking = args.immunity_linking
use_ED_visits = args.use_ED_visits
informed = args.informed
hyperparameters = args.hyperparameters
season = args.season

##############
## Settings ##
##############

# model settings
fips_state = 37
season_start = int(season[0:4])                     # start of season
start_simulation = datetime(season_start, 10, 1)    # date simulation is started

# optimization parameters
## dates
start_calibration = datetime(season_start+1, 4, 25)           # incremental calibration will start from here
end_calibration = datetime(season_start+1, 5, 1)            # and incrementally (weekly) calibrate until this date
end_validation = datetime(season_start+1, 5, 1)             # enddate used on plots
## frequentist optimization
n_pso = 1000                                                # Number of PSO iterations
multiplier_pso = 10                                         # PSO swarm size
## bayesian inference
n_mcmc = 10000                                              # Number of MCMC iterations
multiplier_mcmc = 5                                         # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 10000                                              # Print diagnostics every `print_n`` iterations
discard = 8000                                             # Discard first `discard` iterations as burn-in
thin = 50                                                 # Thinning factor emcee chains
processes = int(os.environ.get('NUM_CORES', '16'))          # Number of CPUs to use
n = 200                                                     # Number of simulations performed in MCMC goodness-of-fit figure

# format model name
if strains:
    model_name = 'SIR-2S'
else:
    model_name = 'SIR-1S'

# calibration parameters
if not immunity_linking:
    pars = ['rho_i', 'T_h', 'rho_h', 'f_R', 'f_I', 'beta', 'delta_beta_temporal']                                       # parameters to calibrate
    bounds = [(1e-4,0.10), (0.5, 14), (1e-4,0.01), (0.01,0.70), (1e-7,1e-3), (0.01,1), (-0.50,0.50)]                    # parameter bounds
    labels = [r'$\rho_{i}$', r'$T_h$', r'$\rho_{h}$',  r'$f_{R}$', r'$f_{I}$', r'$\beta$', r'$\Delta \beta_{t}$']       # labels in output figures
    # UNINFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if not informed:
        # assign priors (R0 ~ N(1.6, 0.2); all other: uninformative)
        log_prior_prob_fcn = 5*[log_prior_uniform,] + 2*[log_prior_normal,]                                                                                   # prior probability functions
        log_prior_prob_fcn_args = [{'bounds':  bounds[0]}, {'bounds':  bounds[1]}, {'bounds':  bounds[2]}, {'bounds':  bounds[3]}, {'bounds':  bounds[4]},
                                    {'avg':  0.455, 'stdev': 0.057}, {'avg':  0, 'stdev': 0.15}]   # arguments prior functions
    # INFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    else:
        # load and select priors
        priors = pd.read_csv('../data/interim/calibration/hyperparameters.csv')
        priors = priors.loc[((priors['model'] == model_name) & (priors['immunity_linking'] == immunity_linking) & (priors['use_ED_visits'] == use_ED_visits)), (['parameter', f'{hyperparameters}'])].set_index('parameter').squeeze()
        # assign values
        if not strains:
            log_prior_prob_fcn = 3*[log_prior_gamma,] + 1*[log_prior_beta,] + 1*[log_prior_gamma,] + 1*[log_prior_normal,] + 12*[log_prior_normal,] 
            log_prior_prob_fcn_args = [ 
                                    # ED visits
                                    {'a': priors['rho_i_a'], 'loc': 0, 'scale': priors['rho_i_scale']},                             # rho_i
                                    {'a': 1, 'loc': 0, 'scale': priors['T_h_scale']},                                               # T_h
                                    # >>>>>>>>>
                                    {'a': priors['rho_h_a'], 'loc': 0, 'scale': priors['rho_h_scale']},                             # rho_h
                                    {'a': priors['f_R_a'], 'b': priors['f_R_b'], 'loc': 0, 'scale': 1},                             # f_R
                                    {'a': priors['f_I_a'], 'loc': 0, 'scale': priors['f_I_scale']},                                 # f_I
                                    {'avg': priors['beta_mu'], 'stdev': priors['beta_sigma']},                                      # beta
                                    {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                    {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
                                    {'avg': priors['delta_beta_temporal_mu_2'], 'stdev': priors['delta_beta_temporal_sigma_2']},
                                    {'avg': priors['delta_beta_temporal_mu_3'], 'stdev': priors['delta_beta_temporal_sigma_3']},
                                    {'avg': priors['delta_beta_temporal_mu_4'], 'stdev': priors['delta_beta_temporal_sigma_4']},
                                    {'avg': priors['delta_beta_temporal_mu_5'], 'stdev': priors['delta_beta_temporal_sigma_5']},
                                    {'avg': priors['delta_beta_temporal_mu_6'], 'stdev': priors['delta_beta_temporal_sigma_6']},
                                    {'avg': priors['delta_beta_temporal_mu_7'], 'stdev': priors['delta_beta_temporal_sigma_7']},
                                    {'avg': priors['delta_beta_temporal_mu_8'], 'stdev': priors['delta_beta_temporal_sigma_8']},
                                    {'avg': priors['delta_beta_temporal_mu_9'], 'stdev': priors['delta_beta_temporal_sigma_9']},
                                    {'avg': priors['delta_beta_temporal_mu_10'], 'stdev': priors['delta_beta_temporal_sigma_10']},
                                    {'avg': priors['delta_beta_temporal_mu_11'], 'stdev': priors['delta_beta_temporal_sigma_11']},
                                    ]          # arguments of prior functions
        else:
            log_prior_prob_fcn = 4*[log_prior_gamma,] + 2*[log_prior_beta,] + 2*[log_prior_gamma,] + 2*[log_prior_normal,] + 12*[log_prior_normal,] 
            log_prior_prob_fcn_args = [ 
                                    # ED visits
                                    {'a': priors['rho_i_a'], 'loc': 0, 'scale': priors['rho_i_scale']},                             # rho_i
                                    {'a': 1, 'loc': 0, 'scale': priors['T_h_scale']},                                               # T_h
                                    # >>>>>>>>>
                                    {'a': priors['rho_h_a_0'], 'loc': 0, 'scale': priors['rho_h_scale_0']},                         # rho_h_0
                                    {'a': priors['rho_h_a_1'], 'loc': 0, 'scale': priors['rho_h_scale_1']},                         # rho_h_1
                                    {'a': priors['f_R_a_0'], 'b': priors['f_R_b_0'], 'loc': 0, 'scale': 1},                         # f_R_0
                                    {'a': priors['f_R_a_1'], 'b': priors['f_R_b_1'], 'loc': 0, 'scale': 1},                         # f_R_1
                                    {'a': priors['f_I_a_0'], 'loc': 0, 'scale': priors['f_I_scale_0']},                             # f_I_0
                                    {'a': priors['f_I_a_1'], 'loc': 0, 'scale': priors['f_I_scale_1']},                             # f_I_1
                                    {'avg': priors['beta_mu_0'], 'stdev': priors['beta_sigma_0']},                                  # beta_0
                                    {'avg': priors['beta_mu_1'], 'stdev': priors['beta_sigma_1']},                                  # beta_1
                                    {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                    {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
                                    {'avg': priors['delta_beta_temporal_mu_2'], 'stdev': priors['delta_beta_temporal_sigma_2']},
                                    {'avg': priors['delta_beta_temporal_mu_3'], 'stdev': priors['delta_beta_temporal_sigma_3']},
                                    {'avg': priors['delta_beta_temporal_mu_4'], 'stdev': priors['delta_beta_temporal_sigma_4']},
                                    {'avg': priors['delta_beta_temporal_mu_5'], 'stdev': priors['delta_beta_temporal_sigma_5']},
                                    {'avg': priors['delta_beta_temporal_mu_6'], 'stdev': priors['delta_beta_temporal_sigma_6']},
                                    {'avg': priors['delta_beta_temporal_mu_7'], 'stdev': priors['delta_beta_temporal_sigma_7']},
                                    {'avg': priors['delta_beta_temporal_mu_8'], 'stdev': priors['delta_beta_temporal_sigma_8']},
                                    {'avg': priors['delta_beta_temporal_mu_9'], 'stdev': priors['delta_beta_temporal_sigma_9']},
                                    {'avg': priors['delta_beta_temporal_mu_10'], 'stdev': priors['delta_beta_temporal_sigma_10']},
                                    {'avg': priors['delta_beta_temporal_mu_11'], 'stdev': priors['delta_beta_temporal_sigma_11']},
                                    ]          # arguments of prior functions
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

else:
    pars = ['rho_i', 'T_h', 'rho_h', 'iota_1', 'iota_2', 'iota_3', 'f_I', 'beta', 'delta_beta_temporal']                                            # parameters to calibrate
    bounds = [(1e-4,0.10), (0.5, 14), (1e-4,0.01), (0,2e-3), (0,2e-3), (0,2e-3), (1e-7,1e-3), (0.01,1), (-0.50,0.50)]                               # parameter bounds
    labels = [r'$\rho_{i}$', r'$T_h$', r'$\rho_{h}$',  r'$\iota_1$', r'$\iota_2$', r'$\iota_3$', r'$f_{I}$', r'$\beta$', r'$\Delta \beta_{t}$']     # labels in output figures
    # UNINFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if not informed:
        # assign priors (R0 ~ N(1.6, 0.2); all other: uninformative)
        log_prior_prob_fcn = 3*[log_prior_uniform,] + 3*[log_prior_gamma] + 1*[log_prior_uniform,] + 2*[log_prior_normal,]                                                                                   # prior probability functions
        log_prior_prob_fcn_args = [{'bounds':  bounds[0]}, {'bounds':  bounds[1]}, {'bounds':  bounds[2]},
                                   {'a': 1, 'loc': 0, 'scale': 2E-04}, {'a': 1, 'loc': 0, 'scale': 2E-04}, {'a': 1, 'loc': 0, 'scale': 2E-04},
                                   {'bounds':  bounds[6]},
                                   {'avg':  0.455, 'stdev': 0.055}, {'avg':  0, 'stdev': 0.15}]   # arguments prior functions
    # INFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    else:
        # load and select priors
        priors = pd.read_csv('../data/interim/calibration/hyperparameters.csv')
        priors = priors.loc[((priors['model'] == model_name) & (priors['immunity_linking'] == immunity_linking) & (priors['use_ED_visits'] == use_ED_visits)), (['parameter', f'{hyperparameters}'])].set_index('parameter').squeeze()
        # assign values
        if not strains:
            log_prior_prob_fcn = 7*[log_prior_gamma,] + 13*[log_prior_normal,] 
            log_prior_prob_fcn_args = [ 
                                    # ED visits
                                    {'a': priors['rho_i_a'], 'loc': 0, 'scale': priors['rho_i_scale']},                             # rho_i
                                    {'a': 1, 'loc': 0, 'scale': priors['T_h_scale']},                                               # T_h
                                    # >>>>>>>>>
                                    {'a': priors['rho_h_a'], 'loc': 0, 'scale': priors['rho_h_scale']},                             # rho_h
                                    {'a': priors['iota_1_a'], 'loc': 0, 'scale': priors['iota_1_scale']},                           # iota_1
                                    {'a': priors['iota_2_a'], 'loc': 0, 'scale': priors['iota_2_scale']},                           # iota_2
                                    {'a': priors['iota_3_a'], 'loc': 0, 'scale': priors['iota_3_scale']},                           # iota_3
                                    {'a': priors['f_I_a'], 'loc': 0, 'scale': priors['f_I_scale']},                                 # f_I
                                    {'avg': priors['beta_mu'], 'stdev': priors['beta_sigma']},                                      # beta
                                    {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                    {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
                                    {'avg': priors['delta_beta_temporal_mu_2'], 'stdev': priors['delta_beta_temporal_sigma_2']},
                                    {'avg': priors['delta_beta_temporal_mu_3'], 'stdev': priors['delta_beta_temporal_sigma_3']},
                                    {'avg': priors['delta_beta_temporal_mu_4'], 'stdev': priors['delta_beta_temporal_sigma_4']},
                                    {'avg': priors['delta_beta_temporal_mu_5'], 'stdev': priors['delta_beta_temporal_sigma_5']},
                                    {'avg': priors['delta_beta_temporal_mu_6'], 'stdev': priors['delta_beta_temporal_sigma_6']},
                                    {'avg': priors['delta_beta_temporal_mu_7'], 'stdev': priors['delta_beta_temporal_sigma_7']},
                                    {'avg': priors['delta_beta_temporal_mu_8'], 'stdev': priors['delta_beta_temporal_sigma_8']},
                                    {'avg': priors['delta_beta_temporal_mu_9'], 'stdev': priors['delta_beta_temporal_sigma_9']},
                                    {'avg': priors['delta_beta_temporal_mu_10'], 'stdev': priors['delta_beta_temporal_sigma_10']},
                                    {'avg': priors['delta_beta_temporal_mu_11'], 'stdev': priors['delta_beta_temporal_sigma_11']},
                                    ]          # arguments of prior functions
        else:
            log_prior_prob_fcn = 12*[log_prior_gamma,] + 14*[log_prior_normal,]
            log_prior_prob_fcn_args = [ 
                                    # ED visits
                                    {'a': priors['rho_i_a'], 'loc': 0, 'scale': priors['rho_i_scale']},                             # rho_i
                                    {'a': 1, 'loc': 0, 'scale': priors['T_h_scale']},                                               # T_h
                                    # >>>>>>>>>
                                    {'a': priors['rho_h_a_0'], 'loc': 0, 'scale': priors['rho_h_scale_0']},                         # rho_h_0
                                    {'a': priors['rho_h_a_1'], 'loc': 0, 'scale': priors['rho_h_scale_1']},                         # rho_h_1
                                    {'a': priors['iota_1_a_0'], 'loc': 0, 'scale': priors['iota_1_scale_0']},                       # iota_1_0
                                    {'a': priors['iota_1_a_1'], 'loc': 0, 'scale': priors['iota_1_scale_1']},                       # iota_1_1
                                    {'a': priors['iota_2_a_0'], 'loc': 0, 'scale': priors['iota_2_scale_0']},                       # iota_2_0
                                    {'a': priors['iota_2_a_1'], 'loc': 0, 'scale': priors['iota_2_scale_1']},                       # iota_2_1
                                    {'a': priors['iota_3_a_0'], 'loc': 0, 'scale': priors['iota_3_scale_0']},                       # iota_3_0
                                    {'a': priors['iota_3_a_1'], 'loc': 0, 'scale': priors['iota_3_scale_1']},                       # iota_3_1
                                    {'a': priors['f_I_a_0'], 'loc': 0, 'scale': priors['f_I_scale_0']},                             # f_I_0
                                    {'a': priors['f_I_a_1'], 'loc': 0, 'scale': priors['f_I_scale_1']},                             # f_I_1
                                    {'avg': priors['beta_mu_0'], 'stdev': priors['beta_sigma_0']},                                  # beta_0
                                    {'avg': priors['beta_mu_1'], 'stdev': priors['beta_sigma_1']},                                  # beta_1
                                    {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                    {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
                                    {'avg': priors['delta_beta_temporal_mu_2'], 'stdev': priors['delta_beta_temporal_sigma_2']},
                                    {'avg': priors['delta_beta_temporal_mu_3'], 'stdev': priors['delta_beta_temporal_sigma_3']},
                                    {'avg': priors['delta_beta_temporal_mu_4'], 'stdev': priors['delta_beta_temporal_sigma_4']},
                                    {'avg': priors['delta_beta_temporal_mu_5'], 'stdev': priors['delta_beta_temporal_sigma_5']},
                                    {'avg': priors['delta_beta_temporal_mu_6'], 'stdev': priors['delta_beta_temporal_sigma_6']},
                                    {'avg': priors['delta_beta_temporal_mu_7'], 'stdev': priors['delta_beta_temporal_sigma_7']},
                                    {'avg': priors['delta_beta_temporal_mu_8'], 'stdev': priors['delta_beta_temporal_sigma_8']},
                                    {'avg': priors['delta_beta_temporal_mu_9'], 'stdev': priors['delta_beta_temporal_sigma_9']},
                                    {'avg': priors['delta_beta_temporal_mu_10'], 'stdev': priors['delta_beta_temporal_sigma_10']},
                                    {'avg': priors['delta_beta_temporal_mu_11'], 'stdev': priors['delta_beta_temporal_sigma_11']},
                                    ]          # arguments of prior functions
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


## starting guestimate NM:
theta = list(pd.read_csv('../data/interim/calibration/single-season-optimal-parameters.csv', index_col=[0,1,2]).loc[(model_name, immunity_linking, slice(None))].mean(axis=1))

##########################################
## Prepare pySODM llp dataset arguments ##
##########################################

data, states, log_likelihood_fnc, log_likelihood_fnc_args = make_data_pySODM_compatible(strains, use_ED_visits, start_simulation, end_calibration, season)

#################
## Setup model ##
#################

model = initialise_model(strains=strains, immunity_linking=immunity_linking, season=season, fips_state=fips_state)

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #####################
    ## Loop over weeks ##
    #####################


    # compute the list of incremental calibration enddates between start_calibration and end_calibration
    incremental_enddates = data[0].loc[slice(start_calibration, end_calibration)].index.get_level_values('date').unique()

    for end_date in incremental_enddates:
        
        print(f"Working on calibration ending on {end_date.strftime('%Y-%m-%d')}, HubVerse reference date: {(end_date+timedelta(weeks=1)).strftime('%Y-%m-%d')}")

        # Make folder structure
        identifier = f'reference_date-{(end_date+timedelta(weeks=1)).strftime('%Y-%m-%d')}' # identifier
        samples_path=fig_path=f'../data/interim/calibration/incremental-calibration/{model_name}/immunity_linking-{immunity_linking}/informed-{informed}_{hyperparameters}/ED_visits-{use_ED_visits}/{season}/{identifier}/' # Path to backend
        run_date = datetime.today().strftime("%Y-%m-%d") # get current date
        # check if samples folder exists, if not, make it
        if not os.path.exists(samples_path):
            os.makedirs(samples_path)

        ##################################
        ## Set up posterior probability ##
        ##################################

        # split data in calibration and validation dataset
        data_calib = [df.loc[slice(start_simulation, end_date)] for df in data]
        data_valid = [df.loc[slice(end_date+timedelta(days=1), end_validation)] for df in data]

        # normalisation weights for lpp
        weights = [1/max(df) for df in data]
        weights = np.array(weights) / np.mean(weights)

        # Setup objective function (no priors defined = uniform priors based on bounds)
        lpp = log_posterior_probability(model, pars, bounds, data_calib, states, log_likelihood_fnc, log_likelihood_fnc_args,
                                                        log_prior_prob_fnc=log_prior_prob_fcn, log_prior_prob_fnc_args=log_prior_prob_fcn_args,
                                                        start_sim=start_simulation, weights=weights, labels=labels)

        #################
        ## Nelder-Mead ##
        #################

        # perform optimization 
        theta, _ = nelder_mead.optimize(lpp, np.array(theta), len(lpp.expanded_bounds)*[0.1,],
                                        processes=processes, max_iter=n_pso, no_improv_break=1000)

        ######################
        ## Visualize result ##
        ######################

        # Assign results to model
        model.parameters = assign_theta(model.parameters, pars, theta)
        # Simulate model
        simout = model.sim([start_simulation, end_validation])
        # visualise output
        plot_fit(simout, data_calib, data_valid, states, fig_path, identifier,
                lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data)

        ##########
        ## MCMC ##
        ##########

        # Perturbate previously obtained estimate
        ndim, nwalkers, pos = perturbate_theta(theta, pert=0.1*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=lpp.expanded_bounds)
        # Append some usefull settings to the samples dictionary
        settings={'start_simulation': start_simulation.strftime('%Y-%m-%d'), 'start_calibration': start_calibration.strftime('%Y-%m-%d'), 'end_calibration': end_date.strftime('%Y-%m-%d'),
                  'season': season, 'starting_estimate': theta}
        # Sample n_mcmc iterations
        sampler, samples_xr = run_EnsembleSampler(pos, n_mcmc, identifier, lpp, fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True, 
                                                    moves=[(emcee.moves.DEMove(), 0.5*0.9),(emcee.moves.DEMove(gamma0=1.0), 0.5*0.1), (emcee.moves.StretchMove(live_dangerously=True), 0.50)],
                                                    settings_dict=settings, discard=discard, thin=thin,
                                            )                                                                               
 
        #######################
        ## Visualize results ##
        #######################

        # Define draw function
        def draw_function(parameters, samples_xr, parameter_shapes):
            """
            A compatible draw function
            """

            # get a random iteration and markov chain
            i = random.randint(0, len(samples_xr.coords['iteration'])-1)
            j = random.randint(0, len(samples_xr.coords['chain'])-1)
            # assign parameters
            for par in parameter_shapes.keys():
                try:
                    if ((par != 'delta_beta_temporal') & (parameter_shapes[par] == (1,))):
                        parameters[par] = np.array([samples_xr[par].sel({'iteration': i, 'chain': j}).values],)
                    else:
                        parameters[par] = samples_xr[par].sel({'iteration': i, 'chain': j}).values
                except:
                    pass
            return parameters

        # Simulate model
        simout = model.sim([start_simulation, end_validation], N=n,
                            draw_function=draw_function, draw_function_kwargs={'samples_xr': samples_xr, 'parameter_shapes': lpp.parameter_shapes})
        
        # Add sampling noise
        try:
            simout = add_poisson_noise(simout)
        except:
            print('no poisson resampling performed')
            sys.stdout.flush()
            pass

        # Save as a .csv in hubverse format / raw netcdf
        df = pySODM_to_hubverse(simout, fips_state, end_date+timedelta(weeks=1), 'wk inc flu hosp', 'H_inc', samples_path, quantiles=True)
        simout.to_netcdf(samples_path+f'{identifier}_simulation-output.nc')

        # Visualise goodnes-of-fit
        plot_fit(simout, data_calib, data_valid, states, fig_path, identifier,
                lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data)
        

