"""
This script makes the 4-week ahead forecast of the influenza model starting from the most recent NHSN HSN data
"""

__author__      = "T.W. Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group (JHUBSPH) & Bento Lab (Cornell CVM). All Rights Reserved."

import sys,os
import argparse
import random
import emcee
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import timedelta
from datetime import datetime as datetime
# pySODM functions
from pySODM.optimization import nelder_mead
from pySODM.optimization.utils import assign_theta, add_poisson_noise
from pySODM.optimization.objective_functions import log_posterior_probability
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler
# hierarchSIR functions
from hierarchSIR.utils import initialise_model, simout_to_hubverse, plot_fit, make_data_pySODM_compatible, get_priors, str_to_bool, samples_to_csv

##############
## Settings ##
##############

# skip fips_state
skip_fips = []

# define hyperparameters to use
hyperparameters = 'exclude_None'

# forecast settings/ save settings
horizon = 4
quantiles = False           # save quantiles vs. individual trajectories 
start_calibration_month = 9       

# optimization parameters
## frequentist optimization
n_nm = 500                                                     # Number of NM search iterations
## bayesian inference
n_mcmc = 2000                                                   # Number of MCMC iterations
multiplier_mcmc = 3                                             # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 2000                                                  # Print diagnostics every `print_n`` iterations
discard = 1500                                                  # Discard first `discard` iterations as burn-in
thin = 10                                                      # Thinning factor emcee chains
processes = int(os.environ.get('NUM_CORES', mp.cpu_count()))    # Number of CPUs to use
n = 2000                                                        # Number of simulations performed in MCMC goodness-of-fit figure

# figure out what states to loop over
initial_guesses = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/interim/calibration/initial_guesses.csv'), index_col=[0,1,2,3,4])
fips_state_list = initial_guesses.index.get_level_values('fips_state').unique().to_list()
fips_state_list = [x for x in fips_state_list if x not in skip_fips]
fips_mappings = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/interim/demography/demography.csv'), dtype={'fips_state': int})
name_state_list = [fips_mappings.loc[fips_mappings['fips_state'] == x]['abbreviation_state'].squeeze() for x in fips_state_list]

# get the latest data (dummy)
data, _, _, _ = make_data_pySODM_compatible(datetime(2000,1,1), datetime(2025,2,1), 1)
end_date = max(data[0].index)

# helper function
def get_influenza_season_label(date: datetime) -> str:
    """
    Given a datetime, return the influenza season label in the format 'YYYY-YYYY'.
    Season runs from September 1 to August 31.
    """
    year = date.year
    if date.month >= 9:  # September or later → start of new season
        start_year = year
        end_year = year + 1
    else:  # January–August → still in previous season
        start_year = year - 1
        end_year = year
    return f"{start_year}-{end_year}"
season = get_influenza_season_label(end_date)

#####################
## Parse arguments ##
#####################

# 'strain' arugment determines the model used in this script
parser = argparse.ArgumentParser()
parser.add_argument("--strains", type=int, default=1, help="Number of strains. Valid options are: 1, 2 (flu A, B) or 3 (flu AH1, AH3, B).")
args = parser.parse_args()
assert args.strains==1, "only valid number of strains is 1."
strains = args.strains

# format number of strains and model name
model_name = f'SIR-{strains}S'

##############
## Let's go ##
##############

# Needed for multiprocessing to work properly
if __name__ == '__main__':

    # Loop over US states
    forecasts = []
    for name_state, fips_state in zip(name_state_list, fips_state_list):

        print(f'Working on forecast in US state {name_state} with hyperparameters {hyperparameters}')
        sys.stdout.flush()

        # optimization parameters
        ## dates
        season_start = int(season[0:4])                                         # start year of season
        start_simulation = datetime(season_start, start_calibration_month, 1)   # date forward simulation is started

        ##########################################
        ## Prepare pySODM llp dataset arguments ##
        ##########################################

        # set up priors
        pars, bounds, labels, log_prior_prob_fcn, log_prior_prob_fcn_args = get_priors(model_name, fips_state, hyperparameters)

        # retrieve guestimate NM
        theta = list(pd.read_csv('../../data/interim/calibration/initial_guesses.csv', index_col=[0,1,2,3,4]).loc[model_name, fips_state, slice(None), slice(None), slice(None)].mean(axis=1))

        # format data
        data, states, log_likelihood_fnc, log_likelihood_fnc_args = make_data_pySODM_compatible(start_simulation, datetime(3000, 1, 1), fips_state)

        # compute relevant dates
        start_simulation = datetime(season_start, start_calibration_month, 1)   # date forward simulation is started
        end_validation = end_date + timedelta(weeks=horizon)

        #################
        ## Setup model ##
        #################

        model = initialise_model(strains=strains, fips_state=fips_state)

        #######################
        ## Start calibration ##
        #######################

        # Make folder structure
        identifier = f'{fips_state}-{name_state}_reference_date-{(end_date+timedelta(weeks=1)).strftime('%Y-%m-%d')}' # identifier
        samples_path=fig_path=f'../../data/interim/calibration/forecast/{model_name}/hyperparameters-{hyperparameters}/reference_date-{(end_date+timedelta(weeks=1)).strftime('%Y-%m-%d')}/{fips_state}-{name_state}/' # Path to backend
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
        if strains > 1:
            weights = [1/max(df) for df in data_calib[:-1]]
            weights = np.array(weights) / np.mean(weights)
            weights = np.append(weights, max(weights))
        else:
            weights = [1/max(df) for df in data_calib]
            weights = np.array(weights) / np.mean(weights)
        

        # Setup objective function (no priors defined = uniform priors based on bounds)
        lpp = log_posterior_probability(model, pars, bounds, data_calib, states, log_likelihood_fnc, log_likelihood_fnc_args,
                                                        log_prior_prob_fnc=log_prior_prob_fcn, log_prior_prob_fnc_args=log_prior_prob_fcn_args,
                                                        start_sim=start_simulation, weights=weights, labels=labels)

        #################
        ## Nelder-Mead ##
        #################

        # perform optimization 
        theta, _ = nelder_mead.optimize(lpp, np.array(theta), len(lpp.expanded_bounds)*[0.2,],
                                        processes=processes, max_iter=n_nm, no_improv_break=1000)

        ######################
        ## Visualize result ##
        ######################

        # Assign results to model
        model.parameters = assign_theta(model.parameters, pars, theta)
        # Simulate model
        simout = model.sim([start_simulation, end_validation])
        # visualise output
        plot_fit(simout, data_calib, data_valid, states, fig_path, identifier,
                lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data, name_state)

        ##########
        ## MCMC ##
        ##########

        # Perturbate previously obtained estimate
        ndim, nwalkers, pos = perturbate_theta(theta, pert=0.2*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=lpp.expanded_bounds)
        # Append some usefull settings to the samples dictionary
        settings={'start_simulation': start_simulation.strftime('%Y-%m-%d'), 'end_calibration': end_date.strftime('%Y-%m-%d'),
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
            simout = add_poisson_noise(simout+0.01)
        except:
            print('no poisson resampling performed')
            sys.stdout.flush()
            pass

        # Save as a .csv in hubverse format / raw netcdf
        df = simout_to_hubverse(simout, fips_state, end_date+timedelta(weeks=1), 'wk inc flu hosp', 'H_inc', samples_path, quantiles=quantiles)
        simout.to_netcdf(samples_path+f'{identifier}_simulation-output.nc')
        forecasts.append(df)

        # Visualise goodnes-of-fit
        plot_fit(simout, data_calib, data_valid, states, fig_path, identifier,
                lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data, name_state)
    
    # spit out final result
    forecasts = pd.concat(forecasts, axis=0)
    forecasts = forecasts.drop(columns=["strain_0"]) # TODO: generalise over strains
    forecasts.to_csv(os.path.join(os.path.dirname(__file__), f'../../data/interim/calibration/forecast/{model_name}/hyperparameters-{hyperparameters}/reference_date-{(end_date+timedelta(weeks=1)).strftime('%Y-%m-%d')}/forecast_reference_date-{(end_date+timedelta(weeks=1)).strftime('%Y-%m-%d')}.csv'))