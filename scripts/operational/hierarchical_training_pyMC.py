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
from hierarchSIR.utils import initialise_model, make_data_pySODM_compatible, str_to_bool, get_NC_influenza_data

import pytensor
import pymc as pm
import pytensor.tensor as pt
pytensor.config.cxx = '/usr/bin/clang++'
pytensor.config.on_opt_error = "ignore"

#####################
## Parse arguments ##
#####################

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--strains", type=int, help="Number of strains. Valid options are: 1, 2 (flu A, B) or 3 (flu AH1, AH3, B).")
parser.add_argument("--immunity_linking", type=str_to_bool, help="Use an immunity linking function.")
parser.add_argument("--use_ED_visits", type=str_to_bool, help="Use ED visit data (ILI) in addition to ED admission data (hosp. adm.).")
args = parser.parse_args()

# assign to desired variables
strains = args.strains
immunity_linking = args.immunity_linking
use_ED_visits = args.use_ED_visits

##############
## Settings ##
##############

# model settings
fips_state = 37

# calibration settings
## datasets
identifiers_list = ['exclude_None',]     # identifiers of training datasets
seasons_list = [                                                                                                    # season to include in training
        ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2023-2024', '2024-2025'],
        ]                                                                                                             
start_calibration_month = 10                                                                                        # start calibration on month 10, day 1
end_calibration_month = 5                                                                                           # end calibration on month 5, day 1
run_date = datetime.today().strftime("%Y-%m-%d")
## define number of chains
n_chains = 4
max_n = 50000
pert = 0.05
processes = min(n_chains, int(os.environ.get('NUM_CORES', mp.cpu_count())))


# Loop over trainings
for seasons, identifier in zip(seasons_list, identifiers_list):
    print(f"\nWorking on calibration with ID: {identifier}")
    sys.stdout.flush()
    
    # Make the folder structure to save results
    ## format model name
    model_name = f'SIR-{strains}S'
    ## define samples path
    samples_path=fig_path=f'../../data/interim/calibration/hierarchical-training/{model_name}/immunity_linking-{immunity_linking}/ED_visits-{use_ED_visits}/{identifier}/' # Path to backend
    ## check if samples folder exists, if not, make it
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    ################
    ## Setup data ##
    ################

    # convert to a list of start and enddates (datetime)
    start_calibrations = [datetime(int(season[0:4]), start_calibration_month, 1) for season in seasons]
    end_calibrations = [datetime(int(season[0:4])+1, end_calibration_month, 1) for season in seasons]

    def get_data(use_ED_visits, strains, start_calibrations, n_observations):
        """
        A function formatting the model's input data

        # TODO: add all strains

        output:
        -------
        data --> (n_season, n_variables, n_observations)
        eval_dates --> (n_season,)
        model_states --> (n_variables,)
        model_states_coord --> (n_variables,)
        """

        eval_dates = []
        data = []
        # loop ove rseasons
        for start_calibration in start_calibrations:
            # get the data & trim temporally
            df = get_NC_influenza_data(start_calibration, None).iloc[:n_observations]
            # save the time index (per season)
            eval_dates.append([d.astype('datetime64[ms]').astype('O') for d in df.index.values])
            # arrange data
            data_season = []
            model_states = []
            strain_coords = []
            if strains == 1:
                data_season.append(df['H_inc'].values)
                model_states.append('H_inc')
                strain_coords.append(None)
            # append ED visits
            if use_ED_visits:
                data_season.append(df['I_inc'].values)
                model_states.append('I_inc')
                strain_coords.append(None)
            # stack data to (n_variables, n_observations)
            data.append(np.stack(data_season))
        # stack data to (n_season, n_variables, n_observations)
        data = np.stack(data, axis=0)

        return data, eval_dates, model_states, strain_coords
    
    # get the data
    n_observations = 31
    data, eval_dates, model_states, strain_coords = get_data(use_ED_visits, strains, start_calibrations, n_observations)

    # compute the tempered likelihood weights


    #####################
    ## Setup ODE model ##
    #####################

    ODE_model = initialise_model(strains=strains, immunity_linking=immunity_linking, season='2014-2015', fips_state=fips_state)

    #######################################
    ## Fetch initial guess of parameters ##
    #######################################

    # parameters: get optimal independent fit with weakly informative prior on R0 and immunity
    pars_model_0 = pd.read_csv('../../data/interim/calibration/single-season-optimal-parameters.csv', index_col=[0,1,2])
    pars_0 = list(pars_model_0.loc[(model_name, immunity_linking, slice(None)), seasons].transpose().values.flatten().tolist())

    # hyperparameters: use all seasons included as the default starting point
    hyperpars_0 = pd.read_csv('../../data/interim/calibration/hyperparameters.csv', index_col=[0,1,2,3])
    hyperpars_0 = hyperpars_0.loc[(model_name, immunity_linking, use_ED_visits, slice(None)), 'exclude_None'].values.tolist()

    # combine
    theta_0 = hyperpars_0 + pars_0

    # Generate random perturbations from a normal distribution
    perturbations = np.random.normal(
            loc=1, scale=pert, size=(n_chains, len(theta_0))
        )
    # Apply perturbations to create the 2D array
    pos = np.array(theta_0)[None, :] * perturbations

    #######################
    ## Define pyMC model ##
    #######################

    n_modifiers = len(ODE_model.parameters['delta_beta_temporal'])
    n_strains = strains
    n_seasons = len(seasons)
    within_season_parameter_names = ['rho_i', 'T_h', 'rho_h', 'f_R', 'f_I', 'beta', 'delta_beta_temporal']
    coords = {
        'season': seasons,
        'strains': range(n_strains),
    }

    def flatten_within_season_params(i, params):
        return pt.concatenate([
            pt.flatten(p[i]) for p in params
        ])
    
    import math
    def sim_one_season(theta, parameter_names, model_states, strain_coords, start_date, end_date, eval_dates):

        # unflatten within-season parameters and assign to model
        pos = 0
        for par in parameter_names:
            par_shape = math.prod(ODE_model.parameter_shapes[par])
            ODE_model.parameters[par] = theta[pos:pos+par_shape]
            pos += par_shape

        # run model
        simout = ODE_model.sim([start_date, end_date])

        # interpolate model output at observation times and retain only n_observations timesteps
        interp = simout.interp({'date': eval_dates}, method="linear").sel({'date': eval_dates})

        # determine what states/coords need to be retained
        output = []
        for state, strain_coord in zip(model_states, strain_coords):
            if not strain_coord:
                output.append(interp[state].sum(dim='strain'))
            else:
                output.append(interp[state].sel({'strain': strain_coord}))

        # return output of shape (n_variables, n_observations)
        return np.vstack(output)

    from pytensor.compile.ops import as_op
    @as_op(itypes=[pt.dmatrix], otypes=[pt.dtensor3])
    def pytensor_forward_model_matrix(theta_matrix):
        """
        theta_matrix: shape (n_seasons, n_parameters) --> flattened within-season parameters
        Returns: predictions (n_seasons, n_variables, n_observations)
        """
        outputs = []
        for i, (start, eval_date) in enumerate(zip(start_calibrations, eval_dates)):
            theta = theta_matrix[i, :]
            sim = sim_one_season(theta, within_season_parameter_names, model_states, strain_coords, start, eval_date[-1], eval_date)
            outputs.append(sim)
        return np.stack(outputs, axis=0)


    with pm.Model() as model:

        if use_ED_visits:
            # rho_i
            rho_i_mu = pm.Uniform('rho_i_mu', lower=0, upper=1e-1, initval=0.025)
            rho_i_sigma = pm.HalfNormal('rho_i_sigma', sigma=1/3)
            rho_i = pm.Truncated('rho_i', pm.LogNormal.dist(mu=pt.log(rho_i_mu), sigma=rho_i_sigma, shape=n_seasons), lower=1e-3, upper=1e-1)

            # T_h
            T_h_mu = pm.Uniform('T_h_mu', lower=0, upper=7, initval=3.5)
            T_h_sigma = pm.HalfNormal('T_h_sigma', sigma=1/3)
            T_h = pm.Truncated('T_h', pm.LogNormal.dist(mu=pt.log(T_h_mu), sigma=T_h_sigma, shape=n_seasons), lower=0.1, upper=14)

        # rho_h
        rho_h_mu = pm.Uniform('rho_h_mu', lower=0, upper=1e-2, initval=0.0025)
        rho_h_sigma = pm.HalfNormal('rho_h_sigma', sigma=1/3)
        rho_h = pm.LogNormal('rho_h', mu=np.log(rho_h_mu), sigma=rho_h_sigma, shape=(n_seasons, n_strains))

        # f_R
        f_R_mu = pm.Normal('f_R_mu', mu=0.4, sigma=0.1)
        f_R_sigma = pm.HalfNormal('f_R_sigma', sigma=0.1)
        f_R = pm.Normal('f_R', mu=f_R_mu, sigma=f_R_sigma, shape=(n_seasons, n_strains))

        # f_I
        f_I_mu = pm.Uniform('f_I_mu', lower=0, upper=1e-3, initval=5e-5)
        f_I_sigma = pm.HalfNormal('f_I_sigma', sigma=1/3)
        f_I = pm.LogNormal('f_I', mu=pt.log(f_I_mu), sigma=f_I_sigma, shape=(n_seasons, n_strains))

        # beta
        beta_mu = pm.Normal('beta_mu', mu=0.455, sigma=0.055)
        beta_sigma = pm.HalfNormal('beta_sigma', sigma=0.055)
        beta = pm.Normal('beta', mu=beta_mu, sigma=beta_sigma, shape=(n_seasons, n_strains))

        # delta_beta_temporal (#TODO: replace with AR-GARCH)
        delta_beta_temporal_mu = pm.Normal('delta_beta_temporal_mu', mu=0, sigma=0.10, shape=n_modifiers)
        delta_beta_temporal_sigma = pm.HalfNormal('delta_beta_temporal_sigma', sigma=1/3, shape=n_modifiers)
        delta_beta_temporal = pm.Normal('delta_beta_temporal', mu=delta_beta_temporal_mu, sigma=delta_beta_temporal_sigma, shape=(n_seasons, n_modifiers))

        # simulate ODE model
        if use_ED_visits:
            within_season_parameter_distributions = [rho_i, T_h, rho_h, f_R, f_I, beta, delta_beta_temporal]
        else:
            within_season_parameter_distributions = [rho_h, f_R, f_I, beta, delta_beta_temporal]
        ## flatten within-season parameters and stack into an (n_seasons, n_parameters) matrix
        thetas = []
        for i in range(n_seasons):
            thetas.append(flatten_within_season_params(i, within_season_parameter_distributions))
        theta_matrix = pt.stack(thetas, axis=0)
        ## run simulation model
        model_predictions = pytensor_forward_model_matrix(theta_matrix)

        # compute likelihood
        obs = pm.Poisson('obs', mu=pt.maximum(model_predictions, 1e-3), observed=data)

    with model:
        trace = pm.sample(10, tune=10, target_accept=0.99, chains=n_chains, cores=processes, init='adapt_diag', progressbar=True)

# Traceplot
variables2plot = [
                'rho_h_mu', 'rho_h_sigma', 'rho_h',     # rho_h
                'f_R_mu', 'f_R_sigma', 'f_R',           # f_R
                'f_I_mu', 'f_I_sigma', 'f_I',           # f_I
                'beta_mu', 'beta_sigma', 'beta',        # beta
                'delta_beta_temporal_mu', 'delta_beta_temporal_sigma'
                ]

if use_ED_visits:
    variables2plot.extend(['rho_i_mu', 'rho_i_sigma', 'rho_i', 'T_h_mu', 'T_h_sigma', 'T_h'])
    
# Save traces
import arviz
import matplotlib.pyplot as plt
os.makedirs(f'{samples_path}/trace', exist_ok=True)
for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(f'{samples_path}/trace/trace-{var}.pdf')
    plt.close()
