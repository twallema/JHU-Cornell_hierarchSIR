"""
This script "trains" (= to find the across-season hyperdistributions) of the influenza model using several seasons of historical data.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import sys,os
import emcee
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from multiprocessing import get_context
from hierarchSIR.training import log_posterior_probability, dump_sampler_to_xarray, traceplot, plot_fit, hyperdistributions
from hierarchSIR.utils import initialise_model, make_data_pySODM_compatible, str_to_bool

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
## Settings ##
##############

# skip fips_state
skip_fips = []

# calibration settings
## datasets
identifiers_list = ['exclude_None']     # identifiers of training datasets
seasons_list = [                                                                                                    # season to include in training
        ['2023-2024', '2024-2025'],
        ]                                                                                                             
start_calibration_month = 9                                                                                        # start calibration on month 10, day 1
end_calibration_month = 6                                                                                           # end calibration on month 5, day 1
run_date = datetime.today().strftime("%Y-%m-%d")
## define number of chains
chain_multiplier = 3
max_n = 1000
pert = 0.05
processes = int(os.environ.get('NUM_CORES', mp.cpu_count()))
## printing and postprocessing
print_n = 1000
backend = None
discard = 800
thin = 5

# figure out what states to loop over
initial_guesses = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/interim/calibration/initial_guesses.csv'), index_col=[0,1,2,3,4])
fips_state_list = initial_guesses.index.get_level_values('fips_state').unique().to_list()
fips_state_list = [x for x in fips_state_list if x not in skip_fips]
fips_mappings = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/interim/demography/demography.csv'), dtype={'fips_state': int})
name_state_list = [fips_mappings.loc[fips_mappings['fips_state'] == x]['name_state'].squeeze() for x in fips_state_list]

# Needed for multiprocessing to work properly
if __name__ == '__main__':

    # Loop over US states
    for name_state, fips_state in zip(name_state_list, fips_state_list):
        print(f"\nWorking on US state: {name_state}")
        sys.stdout.flush()
    
        # Loop over trainings
        for seasons, identifier in zip(seasons_list, identifiers_list):
            print(f"\t\nWorking on calibration with ID: {identifier}")
            sys.stdout.flush()
            
            # Make the folder structure to save results
            ## define samples path
            samples_path=fig_path=f'../../data/interim/calibration/hierarchical-training/{model_name}/{fips_state}-{name_state}/{identifier}/' # Path to backend
            ## check if samples folder exists, if not, make it
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
                data, _, _, _ = make_data_pySODM_compatible(start_calibration, end_calibration, fips_state)
                datasets.append(data)

            #################
            ## Setup model ##
            #################

            model = initialise_model(strains=strains, fips_state=fips_state)

            ##########################################
            ## Setup posterior probability function ##
            ##########################################

            # define model parameters to calibrate to every season and their bounds
            # not how we're not cutting out the parameters associated with the ED visit data

            par_names = ['rho_h', 'f_R', 'f_I', 'beta', 'delta_beta_temporal']
            par_bounds = [(0,0.01), (0,1), (0,1e-3), (0.20,0.60), (-0.5,0.5)]
            par_hyperdistributions = ['lognorm', 'norm', 'lognorm', 'norm', 'norm']

            # setup lpp function
            lpp = log_posterior_probability(model, par_names, par_bounds, par_hyperdistributions, datasets, seasons)

            ####################################
            ## Fetch initial guess parameters ##
            ####################################

            # parameters: get optimal independent fit with weakly informative prior on R0 and immunity
            pars_model_0 = pd.read_csv('../../data/interim/calibration/initial_guesses.csv', index_col=[0,1,2,3,4])
            pars_0 = list(pars_model_0.loc[(model_name, fips_state, slice(None), slice(None), slice(None)), seasons].transpose().values.flatten().tolist())

            # hyperparameters: use all seasons included as the default starting point
            hyperpars_0 = pd.read_csv('../../data/interim/calibration/hyperparameters.csv', index_col=[0,1,2])
            hyperpars_0 = hyperpars_0.loc[(model_name, fips_state, slice(None)), 'initial_guess'].values.tolist()

            # combine
            theta_0 = hyperpars_0 + pars_0

            # run with chain multiplier of minimally two
            n_chains = chain_multiplier*len(theta_0)

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
                backend_file = emcee.backends.HDFBackend(samples_path+fn_backend)
            # If user provides an existing backend: continue sampling 
            else:
                try:
                    backend_file = emcee.backends.HDFBackend(samples_path+backend)
                    pos = backend_file.get_chain(discard=discard, thin=thin, flat=False)[-1, ...]
                except:
                    raise FileNotFoundError("backend not found.")    

            # setup and run sampler
            with get_context("spawn").Pool(processes=processes) as pool:
                # setup sampler
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lpp, backend=backend_file, pool=pool,
                                                moves=[(emcee.moves.DEMove(), 0.5*0.9),(emcee.moves.DEMove(gamma0=1.0), 0.5*0.1),
                                                        (emcee.moves.StretchMove(live_dangerously=True), 0.50)]
                                                )
                # sample
                for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True, skip_initial_state_check=True):

                    if sampler.iteration % print_n:
                        continue
                    else:
                        # every print_n steps do..
                        # >>>>>>>>>>>>>>>>>>>>>>>>>

                        # ..dump samples without discarding and generate traceplots
                        samples = dump_sampler_to_xarray(sampler.get_chain(discard=0, thin=thin), samples_path+str(identifier)+'_SAMPLES_'+run_date+'.nc', lpp.hyperpar_shapes, lpp.par_shapes, seasons)
                        traceplot(samples, lpp.par_shapes, lpp.hyperpar_shapes, samples_path, identifier, run_date)

                        # ..dump samples with discarding and generate other results
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
                        # ..generate goodness-of-fit
                        plot_fit(model, datasets, lpp.simtimes, samples, model.parameter_shapes, samples_path, identifier, run_date,
                                    lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data, lpp.corresponding_model_states, name_state)