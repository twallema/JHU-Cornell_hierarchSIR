"""
This script,
1. takes in the optimized initial parameter guess from `~/data/interim/calibration/initial_guesses.csv`
2. computes an initial guess of the parameters of the hyperdistribution 
3. prepares an excel file for the hyperparameters with an 'initial_guess' column
"""

__author__      = "T.W. Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group (JHUBSPH) & Bento Lab (Cornell CVM). All Rights Reserved."

import os
import numpy as np
import pandas as pd

# get data
initial_guesses = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/interim/calibration/initial_guesses.csv'), index_col=[0,1,2,3,4])

# collection
collect_results = []

# loop over model names
for model_name in initial_guesses.index.get_level_values('model').unique().to_list():

    # loop over fips_state
    for fips_state in initial_guesses.loc[model_name, slice(None), slice(None)].index.get_level_values('fips_state').unique().to_list():

        # get parameters, elements and hyperdistributions 
        parameters = initial_guesses.loc[model_name, fips_state, slice(None)].index.get_level_values('parameter').to_list()
        parameters_unique = initial_guesses.loc[model_name, fips_state, slice(None)].index.get_level_values('parameter').unique().values
        elements = initial_guesses.loc[model_name, fips_state, slice(None)].index.get_level_values('element').to_list()
        hyperdistributions = initial_guesses.loc[model_name, fips_state, slice(None)].index.get_level_values('hyperdistribution').to_list()
        # assumes all hyperdist within unique parameter are the same (#TODO: check)
        hyperdistributions_unique = [initial_guesses.loc[model_name, fips_state, par].index.get_level_values('hyperdistribution').unique().values[0] for par in parameters_unique]
        # make the list of hyperparameters
        hyperparameters_names = []
        hyperparameters_values = []
        # loop over the unique parameter/hyperdist combinations
        for par,hyperdist in zip(parameters_unique, hyperdistributions_unique):
            # loop over elements to fill in first hyperparameter
            for i in range(max(initial_guesses.loc[model_name, fips_state, par].index.get_level_values('element'))+1):
                if hyperdist == 'norm':
                    hyperparameters_names.append(f'{par}_mu_{i}')
                    hyperparameters_values.append(  float(np.mean(initial_guesses.loc[model_name, fips_state, par, i].values)) )
                elif hyperdist == 'lognorm':
                    hyperparameters_names.append(f'{par}_s_{i}')
                    log_data = np.log(np.squeeze(initial_guesses.loc[model_name, fips_state, par, i].values))
                    hyperparameters_values.append(float(np.std(log_data)))
                elif hyperdist == 'beta':
                    hyperparameters_names.append(f'{par}_a_{i}')
                    # sample mean and variance
                    m = np.mean(np.squeeze(initial_guesses.loc[model_name, fips_state, par, i].values))
                    v = np.var(np.squeeze(initial_guesses.loc[model_name, fips_state, par, i].values))
                    # method-of-moments estimates
                    k = (m * (1 - m) / v) - 1
                    a = m * k
                    b = (1 - m) * k
                    hyperparameters_values.append(float(a))
            # loop over elements to fill in second hyperparameter (logic hinges on all hyperdistributions having two parameters; which is fine)
            for i in range(max(initial_guesses.loc[model_name, fips_state, par].index.get_level_values('element'))+1):
                if hyperdist == 'norm':
                    hyperparameters_names.append(f'{par}_sigma_{i}')
                    hyperparameters_values.append(  float(np.std(initial_guesses.loc[model_name, fips_state, par, i].values)) )
                elif hyperdist == 'lognorm':
                    hyperparameters_names.append(f'{par}_scale_{i}')
                    log_data = np.log(np.squeeze(initial_guesses.loc[model_name, fips_state, par, i].values))
                    hyperparameters_values.append(float(np.exp(np.mean(log_data))))
                elif hyperdist == 'beta':
                    hyperparameters_names.append(f'{par}_b_{i}')
                    # sample mean and variance
                    m = np.mean(np.squeeze(initial_guesses.loc[model_name, fips_state, par, i].values))
                    v = np.var(np.squeeze(initial_guesses.loc[model_name, fips_state, par, i].values))
                    # method-of-moments estimates
                    k = (m * (1 - m) / v) - 1
                    a = m * k
                    b = (1 - m) * k
                    hyperparameters_values.append(float(b))

        # get results to a pandas series
        multi_index = pd.MultiIndex.from_product([[model_name,], [fips_state,], hyperparameters_names],
                                                names=['model', 'fips_state', 'hyperparameter'])
        collect_results.append(pd.Series(hyperparameters_values, index=multi_index, name='initial_guess').reset_index())

# concatenate all results
hyperparameters = pd.concat(collect_results, axis=0)

# save result
hyperparameters.to_csv(os.path.join(os.path.dirname(__file__), '../../data/interim/calibration/hyperparameters.csv'), index=False)