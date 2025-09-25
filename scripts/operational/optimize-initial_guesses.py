"""
This script,
1. takes in an initial parameter guess from `~/data/interim/calibration/initial_guesses.csv`
2. uses it to optimise the hierarchSIR model's parameters (in every US state)
3. puts it back in the initial parameter guess file
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
# hierarchDENV functions
from hierarchSIR.utils import initialise_model, plot_fit, make_data_pySODM_compatible, get_priors, str_to_bool, samples_to_csv

##############
## Settings ##
##############

# skip fips_state
skip_fips = []

# season length
season_start_month = 9
season_end_month = 6

# optimization parameters
## frequentist optimization
n_nm = 1000                                                     # Number of NM search iterations
## bayesian inference
n_mcmc = 2000                                                   # Number of MCMC iterations
multiplier_mcmc = 3                                             # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 2000                                                  # Print diagnostics every `print_n`` iterations
discard = 1000                                                  # Discard first `discard` iterations as burn-in
thin = 50                                                       # Thinning factor emcee chains
processes = int(os.environ.get('NUM_CORES', mp.cpu_count()))    # Number of CPUs to use
n = 200                                                         # Number of simulations performed in MCMC goodness-of-fit figure

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

##################################
## Figure out what to loop over ##
##################################

# save the original guesses csv --> we will update this on-the-fly in this script
initial_guesses = pd.read_csv('../../data/interim/calibration/initial_guesses.csv', index_col=[0,1,2,3,4])

# get strains and seasons
fips_state_list = initial_guesses.index.get_level_values('fips_state').unique().to_list()
fips_state_list = [x for x in fips_state_list if x not in skip_fips]
season_lst = initial_guesses.columns.to_list()
fips_mappings = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/interim/demography/demography.csv'), dtype={'fips_state': int})
name_state_list = [fips_mappings.loc[fips_mappings['fips_state'] == x]['name_state'].squeeze() for x in fips_state_list]

##################
## Optimization ##
##################

# Needed for multiprocessing to work properly
if __name__ == '__main__':

    # Start the loop over the federative units and seasons
    for name_state, fips_state in zip(name_state_list, fips_state_list):
        for season in season_lst:

            print(f'Working on the {season} season in US state {fips_state}')
            sys.stdout.flush()

            # dates
            season_start_year = int(season[0:4])                                                        # start year of season
            start_calibration = datetime(season_start_year, season_start_month, 1)                      # date at which calibration starts
            end_calibration = datetime(season_start_year+1, season_end_month, 1)                        # date at which calibration ends
            start_simulation = start_calibration

            ##########################################
            ## Prepare pySODM llp dataset arguments ##
            ##########################################

            # set up priors
            pars, bounds, labels, log_prior_prob_fcn, log_prior_prob_fcn_args = get_priors(model_name, fips_state, None)

            # retrieve initial guess from file
            theta = list(pd.read_csv('../../data/interim/calibration/initial_guesses.csv', index_col=[0,1,2]).loc[(model_name, fips_state, slice(None)), season])

            # format data
            data, states, log_likelihood_fnc, log_likelihood_fnc_args = make_data_pySODM_compatible(start_calibration, end_calibration, fips_state)

            #################
            ## Setup model ##
            #################

            model = initialise_model(strains=strains, fips_state=fips_state)

            #####################
            ## Loop over weeks ##
            #####################

            # Make folder structure
            identifier = f'reference_date-{(end_calibration+timedelta(weeks=1)).strftime('%Y-%m-%d')}' # identifier
            samples_path=fig_path=f'../../data/interim/calibration/optimize-initial_guesses/{model_name}/{fips_state}-{name_state}/{season}/{identifier}/' # Path to backend
            run_date = datetime.today().strftime("%Y-%m-%d") # get current date
            # check if samples folder exists, if not, make it
            if not os.path.exists(samples_path):
                os.makedirs(samples_path)

            ##################################
            ## Set up posterior probability ##
            ##################################

            # split data in calibration and validation dataset (freq: monthly, rescaled to daily)
            data_calib = [df.loc[slice(start_calibration, end_calibration)] for df in data]
            data_valid = [df.loc[slice(end_calibration+timedelta(days=1), end_calibration+timedelta(days=2))] for df in data] # make it empty

            # normalisation weights for lpp
            weights = None
            if strains > 1:
                weights = [1/max(df) for df in data_calib[:-1]]
                weights = np.array(weights) / np.mean(weights)
                weights = np.append(weights, max(weights))

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
            simout = model.sim([start_simulation, end_calibration])
            # visualise output
            plot_fit(simout, data_calib, data_valid, states, fig_path, identifier,
                    lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data, name_state)


            ##########
            ## MCMC ##
            ##########

            # Perturbate previously obtained estimate
            ndim, nwalkers, pos = perturbate_theta(theta, pert=0.01*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=lpp.expanded_bounds)
            # Append some usefull settings to the samples dictionary
            settings={'start_simulation': start_simulation.strftime('%Y-%m-%d'), 'start_calibration': start_calibration.strftime('%Y-%m-%d'), 'end_calibration': end_calibration.strftime('%Y-%m-%d'),
                    'season': season, 'starting_estimate': theta}
            # Sample n_mcmc iterations
            sampler, samples_xr = run_EnsembleSampler(pos, n_mcmc, identifier, lpp, fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True, 
                                                        moves=[(emcee.moves.DEMove(), 0.5*0.9),(emcee.moves.DEMove(gamma0=1.0), 0.5*0.1), (emcee.moves.StretchMove(live_dangerously=True), 0.50)],
                                                        settings_dict=settings, discard=discard, thin=thin,
                                                )   
            
            ##################
            ## Save results ##
            ##################       
                                
            # Retrieve median parameter values across chains and iterations & save in the initial guesses file
            initial_guesses.loc[(model_name, fips_state, slice(None), slice(None), slice(None)), season] = samples_to_csv(samples_xr.median(dim=['chain', 'iteration']))['value'].values

            # Save the initial guesses file
            initial_guesses.to_csv('../../data/interim/calibration/initial_guesses.csv')

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
            simout = model.sim([start_simulation, end_calibration], N=n,
                                draw_function=draw_function, draw_function_kwargs={'samples_xr': samples_xr, 'parameter_shapes': lpp.parameter_shapes})
            
            # Add sampling noise
            try:
                simout = add_poisson_noise(simout+0.01)
            except:
                print('no poisson resampling performed')
                sys.stdout.flush()
                pass

            # Visualise goodnes-of-fit
            plot_fit(simout, data_calib, data_valid, states, fig_path, identifier,
                    lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data, name_state)
            

