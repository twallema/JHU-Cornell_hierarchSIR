"""
This script,
1. takes in an initial parameter guess from `~/data/interim/calibration/initial_guesses.csv`
2. uses it to optimise the hierarchDENV model's parameters (in every federative unit)
3. puts it back in the initial parameter guess file
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

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
from hierarchDENV.utils import initialise_model, plot_fit, make_data_pySODM_compatible, get_priors, str_to_bool, samples_to_csv

##############
## Settings ##
##############

# skip UFs
skip_ufs = []

# season length
season_start_month = 9
season_end_month = 9

# optimization parameters
## frequentist optimization
n_nm = 1000                                                     # Number of NM search iterations
## bayesian inference
n_mcmc = 5000                                                   # Number of MCMC iterations
multiplier_mcmc = 3                                             # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 5000                                                  # Print diagnostics every `print_n`` iterations
discard = 4000                                                  # Discard first `discard` iterations as burn-in
thin = 50                                                       # Thinning factor emcee chains
processes = int(os.environ.get('NUM_CORES', mp.cpu_count()))    # Number of CPUs to use
n = 100                                                         # Number of simulations performed in MCMC goodness-of-fit figure
hyperparameters = None

#####################
## Parse arguments ##
#####################

# arguments determine the model + data combo used to forecast
parser = argparse.ArgumentParser()
parser.add_argument("--serotypes", type=str_to_bool, help="Include serotypes. False: 1 strain model, True: 4 strain model.")
args = parser.parse_args()

# assign to desired variables
serotypes = args.serotypes

# format number of strains and model name
strains = 4 if serotypes is True else 1
model_name = f'SIR-{strains}S'

##################################
## Figure out what to loop over ##
##################################

# save the original guesses csv --> we will update this on-the-fly in this script
initial_guesses_save = pd.read_csv('../../data/interim/calibration/initial_guesses.csv', index_col=[0,1,2,3])

# get ufs and seasons
uf_list = initial_guesses_save.index.get_level_values('uf').unique().to_list()
uf_list = [x for x in uf_list if x not in skip_ufs]
season_lst = initial_guesses_save.columns.to_list()

##################
## Optimization ##
##################

# Needed for multiprocessing to work properly
if __name__ == '__main__':

    # Start the loop over the federative units and seasons
    for uf in uf_list:
        for season in season_lst:

            print(f'Working on the {season} season in {uf}')
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
            pars, bounds, labels, log_prior_prob_fcn, log_prior_prob_fcn_args = get_priors(strains, uf, hyperparameters)

            # retrieve initial guess from file
            theta = list(pd.read_csv('../../data/interim/calibration/initial_guesses.csv', index_col=[0,1,2]).loc[(model_name, uf, slice(None)), season])

            # format data
            data, states, log_likelihood_fnc, log_likelihood_fnc_args = make_data_pySODM_compatible(uf, serotypes, start_calibration, end_calibration)

            #################
            ## Setup model ##
            #################

            model = initialise_model(strains=strains, uf=uf)

            #####################
            ## Loop over weeks ##
            #####################

            # Make folder structure
            identifier = f'reference_date-{(end_calibration+timedelta(weeks=1)).strftime('%Y-%m-%d')}' # identifier
            samples_path=fig_path=f'../../data/interim/calibration/optimize-initial_guesses/{model_name}/{uf}/{season}/hyperpars-{hyperparameters}/{identifier}/' # Path to backend
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
                    lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data, rescaling=30)


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
                                
            # Retrieve median parameter values across chains and iterations
            df = samples_to_csv(samples_xr.median(dim=['chain', 'iteration']))

            # Save in the initial guesses file
            initial_guesses_save.loc[(model_name, uf, slice(None), slice(None)), season] = df.values

            # Save the initial guesses file
            initial_guesses_save.to_csv('../../data/interim/calibration/initial_guesses.csv')

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
                simout = add_poisson_noise(simout+0.1)
            except:
                print('no poisson resampling performed')
                sys.stdout.flush()
                pass

            # Visualise goodnes-of-fit
            plot_fit(simout, data_calib, data_valid, states, fig_path, identifier,
                    lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data, rescaling=30)
            

