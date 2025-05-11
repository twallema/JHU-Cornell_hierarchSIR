"""
This script...
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import math
import random
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.stats import expon, beta, norm, gamma, lognorm
from pySODM.optimization.utils import list_to_dict, add_poisson_noise
from pySODM.optimization.objective_functions import ll_poisson, validate_calibrated_parameters, expand_bounds, validate_dataset, create_fake_xarray_output, compare_data_model_coordinates

###########################################
## Define posterior probability function ##
###########################################

class log_posterior_probability():

    def build_par_idx_map(self, shapes):
        """
        Create and return a dictionary that takes the parameter name as key
        and the index slice of the parameter in the theta vector as value.
        """
        c_idx = 0
        par_name_to_idx = {}
        for name, shape in shapes.items():
            par_name_to_idx[name] = slice(c_idx, c_idx+shape[0])
            c_idx += shape[0]
        return par_name_to_idx

    def par_idxs_all_seasons(self, name):
        """
        Create and return a list of slices for each season for the given parameter name.
        """
        all_idxs = []
        for season in range(self.n_seasons):
            offset = self.n_hyperpars
            offset += self.n_pars * season
            idxs = self.par_name_to_idx[name]
            all_idxs.append(slice(offset+idxs.start, offset+idxs.stop))
        return all_idxs

    # Gamma distribution prior
    @staticmethod
    def gamma_logpdf(theta, season, idxs_per_season, a_idxs, scale_idxs):
        return np.sum(gamma.logpdf(
            theta[idxs_per_season[season]],
            loc=0, a=theta[a_idxs], scale=theta[scale_idxs]
        ))

    # Exponential distribution prior
    @staticmethod
    def expon_logpdf(theta, season, idxs_per_season, scale_idxs):
        return np.sum(expon.logpdf(
            theta[idxs_per_season[season]],
            scale=theta[scale_idxs]
        ))

    # Normal distribution prior
    @staticmethod
    def norm_logpdf(theta, season, idxs_per_season, mu_idxs, sigma_idxs):
        return np.sum(norm.logpdf(
            theta[idxs_per_season[season]],
            loc=theta[mu_idxs], scale=theta[sigma_idxs]
        ))

    # Beta distribution prior
    @staticmethod
    def beta_logpdf(theta, season, idxs_per_season, a_idxs, b_idxs):
        return np.sum(beta.logpdf(
            theta[idxs_per_season[season]],
            a=theta[a_idxs], b=theta[b_idxs]
        ))

    # Log-Normal distribution prior
    @staticmethod
    def lognorm_logpdf(theta, season, idxs_per_season, s_idx, scale_idx):
        return np.sum(lognorm.logpdf(
            theta[idxs_per_season[season]],
            s=theta[s_idx], scale=theta[scale_idx]
        ) + expon.logpdf(theta[s_idx], scale=1))
        
    # Hyper priors for global parameters
    @staticmethod
    def norm_hyper_logpdf(theta, idxs, loc, scale):
        return np.sum(norm.logpdf(theta[idxs], loc=loc, scale=scale))

    @staticmethod
    def expon_hyper_logpdf(theta, idxs, scale):
        return np.sum(expon.logpdf(theta[idxs], scale=scale))
    
    @staticmethod
    def delta_beta_temporal_logpdf(theta, idxs, scale):
        return np.sum(expon.logpdf(np.abs(theta[idxs]), scale=scale))


    def build_prior_evaluation(self):
        """
        Build for every parameter (hyper and seasonal) the prior probability evaluation function
        and store them in the object.
        """
        season_prior_lpp_fs = []
        hyper_prior_lpp_fs = []

        for pars_model_name, pars_model_hyperdistribution in zip(self.par_names, self.par_hyperdistributions):
            idxs_per_season = self.par_name_to_idx_per_season[pars_model_name]

            if pars_model_hyperdistribution == 'gamma':
                a_idxs = self.hyper_par_name_to_idx[f'{pars_model_name}_a']
                scale_idxs = self.hyper_par_name_to_idx[f'{pars_model_name}_scale']
                season_prior_lpp_fs.append((self.gamma_logpdf, (idxs_per_season, a_idxs, scale_idxs)))

            elif pars_model_hyperdistribution == 'expon':
                scale_idxs = self.hyper_par_name_to_idx[f'{pars_model_name}_scale']
                season_prior_lpp_fs.append((self.expon_logpdf, (idxs_per_season, scale_idxs)))

            elif pars_model_hyperdistribution == 'norm':
                mu_idxs = self.hyper_par_name_to_idx[f'{pars_model_name}_mu']
                sigma_idxs = self.hyper_par_name_to_idx[f'{pars_model_name}_sigma']
                season_prior_lpp_fs.append((self.norm_logpdf, (idxs_per_season, mu_idxs, sigma_idxs)))

            elif pars_model_hyperdistribution == 'beta':
                a_idxs = self.hyper_par_name_to_idx[f'{pars_model_name}_a']
                b_idxs = self.hyper_par_name_to_idx[f'{pars_model_name}_b']
                season_prior_lpp_fs.append((self.beta_logpdf, (idxs_per_season, a_idxs, b_idxs)))

            elif pars_model_hyperdistribution == 'lognorm':
                s_idx = self.hyper_par_name_to_idx[f'{pars_model_name}_s']
                scale_idx = self.hyper_par_name_to_idx[f'{pars_model_name}_scale']
                season_prior_lpp_fs.append((self.lognorm_logpdf, (idxs_per_season, s_idx, scale_idx)))

            else:
                raise ValueError(f"'{pars_model_hyperdistribution}' is not a valid hyperdistribution.")

        # Hyperdistribution prior: R0 ~ N(0.455, 0.055)
        beta_mu_idxs = self.hyper_par_name_to_idx['beta_mu']
        hyper_prior_lpp_fs.append((self.norm_hyper_logpdf, (beta_mu_idxs, 0.455, 0.055)))

        # Hyperdistribution prior: beta_sigma ~ Exponential(0.055)
        beta_sigma_idxs = self.hyper_par_name_to_idx['beta_sigma']
        hyper_prior_lpp_fs.append((self.expon_hyper_logpdf, (beta_sigma_idxs, 0.055)))

        # Hyperdistribution prior: delta_beta_temporal_mu ~ Exponential(1)
        delta_beta_mu_idxs = self.hyper_par_name_to_idx['delta_beta_temporal_mu']
        delta_beta_scale = np.ones(delta_beta_mu_idxs.stop - delta_beta_mu_idxs.start)
        hyper_prior_lpp_fs.append((self.delta_beta_temporal_logpdf, (delta_beta_mu_idxs, delta_beta_scale)))

        return season_prior_lpp_fs, hyper_prior_lpp_fs


    def build_selection_dictionaries(self, additional_axes_data):
        """
        Build dictionaries to select the right data and model states
        from the model output.
        """
        # build selection dictionaries
        self.selection_dictionaries = []
        for i, additional_axes in enumerate(additional_axes_data):
            season_dicts = []
            for j, axes in enumerate(additional_axes):
                if axes:
                    season_dicts.append({
                        k: self.coordinates_data_also_in_model[i][j][jdx]
                        for jdx, k in enumerate(axes)
                    })
                else:
                    season_dicts.append(None)
            self.selection_dictionaries.append(season_dicts)

    def __init__(self, model, par_names, par_bounds, par_hyperdistributions, datasets, seasons):
        # get the shapes of the model parameters
        self.par_sizes, self.par_shapes = validate_calibrated_parameters(par_names, model.parameters)
        self.n_pars = sum([v[0] for v in self.par_shapes.values()])

        # build the names and shapes of the hyperparameters
        hyperpar_shapes = {}
        for name, shape, hyperdist in zip(self.par_shapes.keys(), self.par_shapes.values(), par_hyperdistributions):
            if hyperdist == 'gamma':
                hyperpar_shapes[f'{name}_a'] = shape
                hyperpar_shapes[f'{name}_scale'] = shape
            elif hyperdist == 'expon':
                hyperpar_shapes[f'{name}_scale'] = shape
            elif hyperdist == 'norm':
                hyperpar_shapes[f'{name}_mu'] = shape
                hyperpar_shapes[f'{name}_sigma'] = shape
            elif hyperdist == 'beta':
                hyperpar_shapes[f'{name}_a'] = shape
                hyperpar_shapes[f'{name}_b'] = shape
            elif hyperdist == 'lognorm':
                hyperpar_shapes[f'{name}_s'] = shape
                hyperpar_shapes[f'{name}_scale'] = shape
            else:
                raise ValueError(f"'{hyperdist}' is not a valid hyperdistribution.")
        self.hyperpar_shapes = hyperpar_shapes
        self.n_hyperpars = sum(value[0] for value in self.hyperpar_shapes.values())

        # get additional axes beside time axis in dataset and model states we want to match
        self.corresponding_model_states = []
        additional_axes_data = []
        for data in datasets:
            states, addaxis = validate_dataset(data)
            self.corresponding_model_states.append(states)
            additional_axes_data.append(addaxis)

        # check that across seasons, the corresponding states are identical (relaxing introduces additional overhead)
        if not all(sublist == self.corresponding_model_states[0] for sublist in self.corresponding_model_states):
            raise ValueError('across seasons, the states you want to match to must be identical')
        else:
            states_data = self.corresponding_model_states[0]

        # Compare data and model dimensions
        ## Create a fake model output
        out = create_fake_xarray_output(model.dimensions_per_state, model.state_coordinates, model.initial_states, 'date')
        ## Learn what needs to be aggregated over
        self.coordinates_data_also_in_model = []
        self.aggregate_over = []
        for data, states, addaxdata in zip(datasets, self.corresponding_model_states, additional_axes_data):
            aggregation_function = len(self.corresponding_model_states[0]) * [None,]
            out_1, out_2 = compare_data_model_coordinates(out, data, states, aggregation_function, addaxdata)
            self.coordinates_data_also_in_model.append(out_1)
            self.aggregate_over.append(out_2)

        # compute start and end of simulation (per season)
        self.simtimes = []
        for data in datasets:
            start_sim = min([df.index.get_level_values('date').unique().min() for df in data]).to_pydatetime() 
            end_sim = max([df.index.get_level_values('date').unique().max() for df in data]).to_pydatetime()
            self.simtimes.append([start_sim, end_sim])

        # compute the lpp weights matrix
        ## pre-allocate
        w = np.ones([len(datasets), len(states_data)])
        ## weigh by inverse maximum in timeseries
        for i, data in enumerate(datasets):
            for j, _ in enumerate(states_data):
                w[i,j] = 1/max(data[j].values)
        ## normalise back to one
        self.w = w/np.mean(w)

        # assign remaining variables
        self.model = model
        self.par_names = par_names
        self.par_hyperdistributions = par_hyperdistributions
        self.datasets = datasets
        self.seasons = seasons
        self.n_seasons = len(seasons)
        # expand the model parameters bounds (beta_temporal is 1D)
        self.pars_model_bounds = expand_bounds(self.par_sizes, par_bounds)
        self.par_name_to_idx = self.build_par_idx_map(self.par_shapes)
        self.par_name_to_idx_per_season = {k: self.par_idxs_all_seasons(k) for k in self.par_name_to_idx.keys()}
        self.hyper_par_name_to_idx = self.build_par_idx_map(self.hyperpar_shapes)
        self.season_prior_lpp_fs, self.hyper_prior_lpp_fs = self.build_prior_evaluation()

        # Extract necessary information from datasets
        self.build_selection_dictionaries(additional_axes_data)
        self.data_xs = [[df.squeeze().values for df in data] for data in datasets]
        self.data_dates = [[df.index.get_level_values('date').unique().values for df in data] for data in datasets]

    def check_bounds(self, theta):
        """
        Check if the parameters are within the bounds defined in self.pars_model_bounds
        """
        non_hyper_pars = theta[self.n_hyperpars:]
        for i, (lb, ub) in enumerate(self.pars_model_bounds):
            for j in range(self.n_seasons):
                if non_hyper_pars[i + j*self.n_pars] < lb or non_hyper_pars[i + j*self.n_pars] > ub:
                    return False
        return True

    def unpack_y(self, out_copy, state_model, dates, i, j):
        """
        Unpack the y data from the model output.
        """
        selection = {'date': dates}
        if self.selection_dictionaries[i][j]:
            selection.update(self.selection_dictionaries[i][j])

        # Single selection call
        result = out_copy[state_model].sel(selection).values

        # Apply squeeze only if needed
        return np.squeeze(result) if self.selection_dictionaries[i][j] else result

    def __call__(self, theta: np.ndarray) -> float:
        """
        Computes the log posterior probability

        input
        -----

        - theta: list
            - estimated parameters

        output
        ------
        - lpp: float
            - associated log posterior probability
        """

        # check if theta is within bounds
        if not self.check_bounds(theta):
            return -np.inf

        # pre-allocate lpp
        lpp = 0

         # compute hyperdistribution priors
        for hyper_prior_lpp, pars in self.hyper_prior_lpp_fs:
            lpp += hyper_prior_lpp(theta, *pars)
        lpp *= self.n_seasons

        # loop over the seasons
        for i, (season, data_x, data_date) in enumerate(zip(self.seasons, self.data_xs, self.data_dates)):

            # compute priors
            for season_prior_lpp, pars in self.season_prior_lpp_fs:
                lpp += season_prior_lpp(theta, i, *pars)

            # negative arguments in hyperparameters lead to a nan lpp --> redact to -np.inf and move on
            if math.isnan(lpp):
                return -np.inf

            # Assign model parameters
            for par in self.model.parameters.keys():
                if not par in self.par_name_to_idx.keys():
                    continue
                self.model.parameters[par] = theta[self.par_name_to_idx_per_season[par][i]]

            # Set the right season (needed for the immunity linking)
            self.model.parameters['season'] = season

            # run the forward simulation
            simout = self.model.sim(self.simtimes[i])

            # loop over states
            for j, (state_model, x, dates) in enumerate(zip(self.corresponding_model_states[i], data_x, data_date)):
                # reset copy
                out_copy = simout
                # aggregate over unwanted dimensions
                for dimension in simout.dims:
                    if dimension in self.aggregate_over[i][j]:
                        out_copy = out_copy.sum(dim=dimension)
                # match data and model
                # get timeseries
                #x = df.squeeze().values
                y = self.unpack_y(out_copy, state_model, dates, i, j)
                # check if shapes are consistent
                if x.shape != y.shape:
                    raise Exception(f"shape of model prediction {y.shape} and data {x.shape} are not identical.")
                # check for nan in model output
                if np.isnan(y).any():
                    raise ValueError(f"simulation output contains nan, most likely due to numerical unstability. try using more conservative bounds.")
                # compute lpp
                lpp += self.w[i,j] * ll_poisson(x, y)

        return lpp

######################
## Helper functions ##
######################

def validate_dataset(data):
    """
    Validates a dataset:
        - Does the dataset itself have the right type?
        - Does it contain Nan's?
        - Is the index level 'date' present? (obligated)
        - Are the indices in 'date' all datetime?


    Extracts and returns the additional dimensions in dataset besides the 'date' axis, as well as the desired model state that will be matched to the data.

    Parameters
    ----------

    data: list
        List containing the datasets (pd.Series, pd.Dataframe, xarray.DataArray, xarray.Dataset)
    
    Returns
    -------

    corresponding_model_states: list
        Contains the name of the pd.Series -- assumed to be the model state that must be matched

    additional_axes_data: list
        Contains the index levels beside 'date' present in the dataset
    """

    corresponding_model_states = []
    additional_axes_data=[] 
    for idx, df in enumerate(data):
        # Is dataset a pd.Series or pd.Dataframe?
        if not isinstance(df, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"{idx}th dataset is of type {type(df)}. expected pd.Series, pd.DataFrame or xarray.DataArray, xarray.Dataset"
            )
        # If it is a pd.DataFrame, does it have one column?
        if isinstance(df, pd.DataFrame):
            if len(df.columns) != 1:
                raise ValueError(
                    f"{idx}th dataset is a pd.DataFrame with {len(df.columns)} columns. expected one column."
                )
            else:
                corresponding_model_states.append(df.columns.values)
        else:
            corresponding_model_states.append(df.name)
        # Does data contain NaN values anywhere?
        if np.isnan(df).values.any():
            raise ValueError(
                f"{idx}th dataset contains nans"
                )
        # Verify dataset is not empty
        assert not df.empty, f"{idx}th dataset is empty"
        # Does data have 'date' or 'time' as index level? (required)
        if 'date' not in df.index.names:
            raise ValueError(
                f"Index of {idx}th dataset does not have 'date' as index level (index levels: {df.index.names})."
                )
        else:
            additional_axes_data.append([name for name in df.index.names if name != 'date'])

    # Do the types of the time axis make sense?
    for idx, df in enumerate(data):
        time_index_vals = df.index.get_level_values('date').unique().values
        # 'date' --> can only be (np.)datetime; no str representations --> gets very messy if we allow this
        if not all([isinstance(t, (datetime, np.datetime64)) for t in time_index_vals]):
            raise ValueError(f"index level 'date' of the {idx}th dataset contains values not of type 'datetime'")

    return corresponding_model_states, additional_axes_data

#################################
## Function to save the chains ##
#################################

def dump_sampler_to_xarray(samples_np, path_filename, hyperpars_shapes, pars_shapes, seasons):
    """
    A function converting the raw samples from `emcee` (numpy matrix) to a more convenient xarray dataset
    """
        
    # split the hyperparameters from the season's parameters
    n_hyperpars = sum(value[0] for value in hyperpars_shapes.values())
    n_pars = sum([v[0] for v in pars_shapes.values()])
    samples_hyperpars = samples_np[:, :, :n_hyperpars]
    samples_pars = samples_np[:, :, n_hyperpars:]

    # format hyperpars
    data = {}
    i=0
    for par, shape in hyperpars_shapes.items():
        # cut out right parameter
        arr = np.squeeze(samples_hyperpars[:,:,i:i+shape[0]])
        # update counter
        i += shape[0]
        # construct dims and coords
        dims = ['iteration', 'chain']
        coords = {'iteration': range(arr.shape[0]), 'chain': range(arr.shape[1])}
        if shape[0] > 1:
            dims.append(f'{par}_dim')
            coords[f'{par}_dim'] = range(shape[0])
        # wrap in an xarray
        data[par] = xr.DataArray(arr, dims=dims, coords=coords)

    # format pars
    i=0
    for par, shape in pars_shapes.items():
        arr = []
        for j, _ in enumerate(seasons):
            # cut out right parameter, season
            arr.append(np.squeeze(samples_pars[:,:,j*n_pars+i:j*n_pars+i+shape[0]]))
        # update counter
        i += shape[0]
        # stack parameter across seasons
        arr = np.stack(arr, axis=-1)
        # construct dims and coords
        dims = ['iteration', 'chain']
        coords = {'iteration': range(arr.shape[0]), 'chain': range(arr.shape[1])}
        if shape[0] > 1:
            dims.append(f'{par}_dim')
            coords[f'{par}_dim'] = range(shape[0])
        dims.append('season')
        coords['season'] = seasons
        # wrap in an xarray
        data[par] = xr.DataArray(arr, dims=dims, coords=coords)

    # combine it all
    samples_xr = xr.Dataset(data)

    # save it
    samples_xr.to_netcdf(path_filename)

    return samples_xr

########################
## Hyperdistributions ##
########################

def hyperdistributions(samples_xr, path_filename, pars_model_shapes, hyperpars_shapes, pars_model_hyperdistributions, bounds, N):

    # get the element-expanded number of parameters and the parameter's names
    pars_model_names = pars_model_shapes.keys()
    
    # compute the size of the figure
    n_subfigs=0
    for par, hyperdistribution in zip(pars_model_names, pars_model_hyperdistributions):
        if par == 'delta_beta_temporal':
            n_subfigs += 1
        else:
            if hyperdistribution == 'gamma':
                ### construct hyperpars names
                a_name = f'{par}_a'
                ### get shapes
                n_subfigs += hyperpars_shapes[a_name][0]
            elif hyperdistribution == 'expon':
                ### construct hyperpars names
                scale_name = f'{par}_scale'
                ### get shapes
                n_subfigs += hyperpars_shapes[scale_name][0]
            elif hyperdistribution == 'norm':
                ### construct hyperpars names
                mu_name = f'{par}_mu'
                ### get shapes
                n_subfigs += hyperpars_shapes[mu_name][0]
            elif hyperdistribution == 'beta':
                ### construct hyperpars names
                a_name = f'{par}_a'
                ### get shapes
                n_subfigs += hyperpars_shapes[a_name][0]
            elif hyperdistribution == 'lognorm':
                ### construct hyperpars names
                s_name = f'{par}_s'
                ### get shapes
                n_subfigs += hyperpars_shapes[s_name][0]

    # compute subfigure shape
    ncols = 2
    nrows = int((n_subfigs - (n_subfigs % ncols))/ncols + 1)
    
    # make figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.3,11.7/5*nrows))

    ax = axes.flatten()

    i = 0
    for _, (par_name, par_hyperdistribution, bound) in enumerate(zip(pars_model_names, pars_model_hyperdistributions, bounds)):

        # exception for `delta_beta_temporal`
        if par_name == 'delta_beta_temporal':

            ### get transmission rate function
            from hierarchSIR.utils import get_transmission_coefficient_timeseries
            ### compute modifier tranjectory of every season and plot
            for _, season in enumerate(samples_xr.coords['season']):
                y = 1 + get_transmission_coefficient_timeseries(samples_xr['delta_beta_temporal'].median(dim=['iteration', 'chain']).sel(season=season).values, sigma=2.5)
                x = pd.date_range(start=datetime(2020,9,15), end=datetime(2020,9,15) + timedelta(days=len(y)-1), freq='D').tolist()
                ax[i].plot(x, y, color='black', linewidth=0.5, alpha=0.2)
            ### visualise hyperdistribution
            ll= 1 + get_transmission_coefficient_timeseries(samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values - samples_xr['delta_beta_temporal_sigma'].median(dim=['iteration', 'chain']).values, sigma=2.5)
            y= 1 + get_transmission_coefficient_timeseries(samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values, sigma=2.5)
            ul=1 + get_transmission_coefficient_timeseries(samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values + samples_xr['delta_beta_temporal_sigma'].median(dim=['iteration', 'chain']).values, sigma=2.5)
            ax[i].plot(x, y, color='red', alpha=0.8)
            ax[i].fill_between(x, ll, ul, color='red', alpha=0.1)
            # add parameter box
            ax[i].text(0.02, 0.97, f"avg={list(np.round(samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values,2).tolist())}\nstdev={list(np.round(samples_xr['delta_beta_temporal_sigma'].median(dim=['iteration', 'chain']).values,2).tolist())}", transform=ax[i].transAxes, fontsize=5,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax[i].set_ylabel(r'$\Delta \beta_{t}$')
            ax[i].set_xlim([datetime(2020,10,21), datetime(2021, 4, 12)])
            ax[i].set_ylim([0.5, 1.5])
            # update 
            i += 1
        else:

            # define x based on plausible range
            x = np.linspace(start=bound[0],stop=bound[1],num=100)

            ## if not `delta_beta_temporal`: select the right distribution
            if par_hyperdistribution == 'gamma':
                ### construct hyperpars names
                a_name = f'{par_name}_a'
                scale_name = f'{par_name}_scale'
                ### get shape
                n_strains = hyperpars_shapes[a_name][0]
                for j in range(n_strains):
                    # loop over the number of iterations
                    for k in range(N):
                        ## draw a random sample
                        m = random.randint(0, len(samples_xr.coords['iteration'])-1)
                        n = random.randint(0, len(samples_xr.coords['chain'])-1)
                        if n_strains == 1:
                            ### add sample to plot
                            ax[i+j].plot(x, gamma.pdf(x, a=samples_xr[a_name].sel({'iteration': m, 'chain': n}).values, scale=samples_xr[scale_name].sel({'iteration': m, 'chain': n}).values), alpha=0.05, color='black')
                            ### add mean
                            if k == N-1:
                                #### mean
                                ax[i+j].plot(x, gamma.pdf(x, a=samples_xr[a_name].median(dim=['iteration', 'chain']).values, scale=samples_xr[scale_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                                #### textbox
                                ax[i+j].text(0.05, 0.95, f"a={samples_xr[a_name].median(dim=['iteration', 'chain']).values:.1e}, scale={samples_xr[scale_name].median(dim=['iteration', 'chain']).values:.1e}", transform=ax[i+j].transAxes, fontsize=7,
                                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                                ax[i+j].set_ylabel(f'{par_name}')
                        else:
                            ### add sample to plot
                            ax[i+j].plot(x, gamma.pdf(x, a=samples_xr[a_name].sel({'iteration': m, 'chain': n, f'{a_name}_dim': j}).values, scale=samples_xr[scale_name].sel({'iteration': m, 'chain': n, f'{scale_name}_dim': j}).values), alpha=0.05, color='black')
                            ### add mean
                            if k == N-1:
                                #### mean
                                ax[i+j].plot(x, gamma.pdf(x, a=samples_xr[a_name].sel({f'{a_name}_dim': j}).median(dim=['iteration', 'chain']).values, scale=samples_xr[scale_name].sel({f'{scale_name}_dim': j}).median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                                #### textbox
                                ax[i+j].text(0.05, 0.95, f"a={samples_xr[a_name].sel({f'{a_name}_dim': j}).median(dim=['iteration', 'chain']).values:.1e}, scale={samples_xr[scale_name].sel({f'{scale_name}_dim': j}).median(dim=['iteration', 'chain']).values:.1e}", transform=ax[i+j].transAxes, fontsize=7,
                                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                                ax[i+j].set_ylabel(f'{par_name}_{j}')
                i+=n_strains

            elif par_hyperdistribution == 'expon':
                ### construct hyperpars names
                scale_name = f'{par_name}_scale'
                ### get shape
                n_strains = hyperpars_shapes[scale_name][0]
                for j in range(n_strains):
                    # loop over the number of iterations
                    for k in range(N):
                        ## draw a random sample
                        m = random.randint(0, len(samples_xr.coords['iteration'])-1)
                        n = random.randint(0, len(samples_xr.coords['chain'])-1)
                        if n_strains == 1:
                            ### add sample to plot
                            ax[i+j].plot(x, expon.pdf(x, scale=samples_xr[scale_name].sel({'iteration': m, 'chain': n}).values), alpha=0.05, color='black')
                            ### add mean
                            if k == N-1:
                                #### mean
                                ax[i+j].plot(x, expon.pdf(x, scale=samples_xr[scale_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                                #### textbox
                                ax[i+j].text(0.05, 0.95, f"scale={samples_xr[scale_name].median(dim=['iteration', 'chain']).values:.1e}", transform=ax[i+j].transAxes, fontsize=7,
                                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                                ax[i+j].set_ylabel(f'{par_name}')
                        else:
                            ### add sample to plot
                            ax[i+j].plot(x, expon.pdf(x, scale=samples_xr[scale_name].sel({'iteration': m, 'chain': n, f'{scale_name}_dim': j}).values), alpha=0.05, color='black')
                            ### add mean
                            if k == N-1:
                                #### mean
                                ax[i+j].plot(x, expon.pdf(x, scale=samples_xr[scale_name].sel({f'{scale_name}_dim': j}).median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                                #### textbox
                                ax[i+j].text(0.05, 0.95, f"scale={samples_xr[scale_name].sel({f'{scale_name}_dim': j}).median(dim=['iteration', 'chain']).values:.1e}", transform=ax[i+j].transAxes, fontsize=7,
                                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                                ax[i+j].set_ylabel(f'{par_name}_{j}')
                i+=n_strains

            elif par_hyperdistribution == 'norm':
                ### construct hyperpars names
                mu_name = f'{par_name}_mu'
                sigma_name = f'{par_name}_sigma'
                ### get shape
                n_strains = hyperpars_shapes[mu_name][0]
                for j in range(n_strains):
                    # loop over the number of iterations
                    for k in range(N):
                        ## draw a random sample
                        m = random.randint(0, len(samples_xr.coords['iteration'])-1)
                        n = random.randint(0, len(samples_xr.coords['chain'])-1)
                        if n_strains == 1:
                            ### add sample to plot
                            ax[i+j].plot(x, norm.pdf(x, loc=samples_xr[mu_name].sel({'iteration': m, 'chain': n}).values, scale=samples_xr[sigma_name].sel({'iteration': m, 'chain': n}).values), alpha=0.05, color='black')
                            ### add mean
                            if k == N-1:
                                #### mean
                                ax[i+j].plot(x, norm.pdf(x, loc=samples_xr[mu_name].median(dim=['iteration', 'chain']).values, scale=samples_xr[sigma_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                                #### textbox
                                ax[i+j].text(0.05, 0.95, f"avg={samples_xr[mu_name].median(dim=['iteration', 'chain']).values:.1e}, stdev={samples_xr[sigma_name].median(dim=['iteration', 'chain']).values:.1e}", transform=ax[i+j].transAxes, fontsize=7,
                                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                                ax[i+j].set_ylabel(f'{par_name}')
                        else:
                            ### add sample to plot
                            ax[i+j].plot(x, norm.pdf(x, loc=samples_xr[mu_name].sel({'iteration': m, 'chain': n, f'{mu_name}_dim': j}).values, scale=samples_xr[sigma_name].sel({'iteration': m, 'chain': n, f'{sigma_name}_dim': j}).values), alpha=0.05, color='black')
                            ### add mean
                            if k == N-1:
                                #### mean
                                ax[i+j].plot(x, norm.pdf(x, loc=samples_xr[mu_name].sel({f'{mu_name}_dim': j}).median(dim=['iteration', 'chain']).values, scale=samples_xr[sigma_name].sel({f'{sigma_name}_dim': j}).median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                                #### textbox
                                ax[i+j].text(0.05, 0.95, f"avg={samples_xr[mu_name].sel({f'{mu_name}_dim': j}).median(dim=['iteration', 'chain']).values:.1e}, stdev={samples_xr[sigma_name].sel({f'{sigma_name}_dim': j}).median(dim=['iteration', 'chain']).values:.1e}", transform=ax[i+j].transAxes, fontsize=7,
                                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                                ax[i+j].set_ylabel(f'{par_name}_{j}')
                i+=n_strains


            elif par_hyperdistribution == 'beta':
                ### construct hyperpars names
                a_name = f'{par_name}_a'
                b_name = f'{par_name}_b'
                ### get shape
                n_strains = hyperpars_shapes[a_name][0]
                for j in range(n_strains):
                    # loop over the number of iterations
                    for k in range(N):
                        ## draw a random sample
                        m = random.randint(0, len(samples_xr.coords['iteration'])-1)
                        n = random.randint(0, len(samples_xr.coords['chain'])-1)
                        if n_strains == 1:
                            ### add sample to plot
                            ax[i+j].plot(x, beta.pdf(x, a=samples_xr[a_name].sel({'iteration': m, 'chain': n}).values, b=samples_xr[b_name].sel({'iteration': m, 'chain': n}).values), alpha=0.05, color='black')
                            ### add mean
                            if k == N-1:
                                #### mean
                                ax[i+j].plot(x, beta.pdf(x, a=samples_xr[a_name].median(dim=['iteration', 'chain']).values, b=samples_xr[b_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                                #### textbox
                                ax[i+j].text(0.05, 0.95, f"a={samples_xr[a_name].median(dim=['iteration', 'chain']).values:.1e}, b={samples_xr[b_name].median(dim=['iteration', 'chain']).values:.1e}", transform=ax[i+j].transAxes, fontsize=7,
                                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                                ax[i+j].set_ylabel(f'{par_name}')
                        else:
                            ### add sample to plot
                            ax[i+j].plot(x, beta.pdf(x, a=samples_xr[a_name].sel({'iteration': m, 'chain': n, f'{a_name}_dim': j}).values, b=samples_xr[b_name].sel({'iteration': m, 'chain': n, f'{b_name}_dim': j}).values), alpha=0.05, color='black')
                            ### add mean
                            if k == N-1:
                                #### mean
                                ax[i+j].plot(x, beta.pdf(x, a=samples_xr[a_name].sel({f'{a_name}_dim': j}).median(dim=['iteration', 'chain']).values, b=samples_xr[b_name].sel({f'{b_name}_dim': j}).median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                                #### textbox
                                ax[i+j].text(0.05, 0.95, f"a={samples_xr[a_name].sel({f'{a_name}_dim': j}).median(dim=['iteration', 'chain']).values:.1e}, b={samples_xr[b_name].sel({f'{b_name}_dim': j}).median(dim=['iteration', 'chain']).values:.1e}", transform=ax[i+j].transAxes, fontsize=7,
                                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                                ax[i+j].set_ylabel(f'{par_name}_{j}')
                i+=n_strains

            elif par_hyperdistribution == 'lognorm':
                ### construct hyperpars names
                s_name = f'{par_name}_s'
                scale_name = f'{par_name}_scale'
                ### get shape
                n_strains = hyperpars_shapes[s_name][0]
                for j in range(n_strains):
                    # loop over the number of iterations
                    for k in range(N):
                        ## draw a random sample
                        m = random.randint(0, len(samples_xr.coords['iteration'])-1)
                        n = random.randint(0, len(samples_xr.coords['chain'])-1)
                        if n_strains == 1:
                            ### add sample to plot
                            ax[i+j].plot(x, lognorm.pdf(x, s=samples_xr[s_name].sel({'iteration': m, 'chain': n}).values, scale=samples_xr[scale_name].sel({'iteration': m, 'chain': n}).values), alpha=0.05, color='black')
                            ### add mean
                            if k == N-1:
                                #### mean
                                ax[i+j].plot(x, lognorm.pdf(x, s=samples_xr[s_name].median(dim=['iteration', 'chain']).values, scale=samples_xr[scale_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                                #### textbox
                                ax[i+j].text(0.05, 0.95, f"s={samples_xr[s_name].median(dim=['iteration', 'chain']).values:.1e}, scale={samples_xr[scale_name].median(dim=['iteration', 'chain']).values:.1e}", transform=ax[i+j].transAxes, fontsize=7,
                                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                                ax[i+j].set_ylabel(f'{par_name}')
                        else:
                            ### add sample to plot
                            ax[i+j].plot(x, lognorm.pdf(x, s=samples_xr[s_name].sel({'iteration': m, 'chain': n, f'{s_name}_dim': j}).values, scale=samples_xr[scale_name].sel({'iteration': m, 'chain': n, f'{scale_name}_dim': j}).values), alpha=0.05, color='black')
                            ### add mean
                            if k == N-1:
                                #### mean
                                ax[i+j].plot(x, lognorm.pdf(x, s=samples_xr[s_name].sel({f'{s_name}_dim': j}).median(dim=['iteration', 'chain']).values, scale=samples_xr[scale_name].sel({f'{scale_name}_dim': j}).median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                                #### textbox
                                ax[i+j].text(0.05, 0.95, f"s={samples_xr[s_name].sel({f'{s_name}_dim': j}).median(dim=['iteration', 'chain']).values:.1e}, scale={samples_xr[scale_name].sel({f'{scale_name}_dim': j}).median(dim=['iteration', 'chain']).values:.1e}", transform=ax[i+j].transAxes, fontsize=7,
                                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                                ax[i+j].set_ylabel(f'{par_name}_{j}')
                i+=n_strains

    plt.tight_layout()
    plt.savefig(path_filename)
    plt.close()

    pass


#####################
## Goodness-of-fit ##
#####################

def plot_fit(model, datasets, simtimes, samples_xr, parameter_shapes, path, identifier, run_date,
                coordinates_data_also_in_model, aggregate_over, additional_axes_data, corresponding_model_states):
    """
    Visualises the goodness of fit for every season
    """

    # define draw function
    def draw_function(parameters, samples_xr, season, parameter_shapes):
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
                    parameters[par] = np.array([samples_xr[par].sel({'iteration': i, 'chain': j, 'season': season}).values],)
                else:
                    parameters[par] = samples_xr[par].sel({'iteration': i, 'chain': j, 'season': season}).values
            except:
                pass
        return parameters

    # LOOP seasons
    simout=[]
    for season, data, simtime in zip(list(samples_xr.coords['season'].values), datasets, simtimes):

        # set the season
        model.parameters['season'] = season

        # simulate model
        out = model.sim(simtime, N=100, draw_function=draw_function,
                        draw_function_kwargs={'samples_xr': samples_xr, 'season': season, 'parameter_shapes': parameter_shapes})
        
        # try poisson resampling
        try:
            out = add_poisson_noise(out)
        except:
            print('no poisson resampling performed')
            pass

        # append result
        simout.append(out)
    
    # LOOP seasons
    for i, (season, data, out) in enumerate(zip(list(samples_xr.coords['season'].values), datasets, simout)):
        
        # compute the amount of timeseries
        nrows = sum(1 if not coords else len(coords) for coords in coordinates_data_also_in_model[i])

        # generate figure
        fig,ax=plt.subplots(nrows=nrows, sharex=True, figsize=(8.3, 11.7/5*nrows))

        # vectorise ax object
        if nrows==1:
            ax = [ax,]

        # save a copy to reset
        out_copy = out

        # loop over datasets
        k=0
        for j, df in enumerate(data):
            
            # aggregate data
            for dimension in out.dims:
                if dimension in aggregate_over[i][j]:
                    out = out.sum(dim=dimension)
            
            # loop over coordinates 
            if coordinates_data_also_in_model[i][j]:
                for coord in coordinates_data_also_in_model[i][j]:
                    # get dimension coord is in #TODO: LIMITED TO ONE COORDINATE PER DIMENSION PER DATASET !!!
                    dim_name = additional_axes_data[i][j][0]
                    coord = coord[0]
                    # plot
                    ax[k].scatter(df.index.get_level_values('date').values, 7*df.loc[slice(None), coord].values, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
                    ax[k].fill_between(out['date'], 7*out[corresponding_model_states[i][j]].sel({dim_name: coord}).quantile(dim='draws', q=0.05/2),
                                7*out[corresponding_model_states[i][j]].sel({dim_name: coord}).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
                    ax[k].fill_between(out['date'], 7*out[corresponding_model_states[i][j]].sel({dim_name: coord}).quantile(dim='draws', q=0.50/2),
                                7*out[corresponding_model_states[i][j]].sel({dim_name: coord}).quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)
                    ax[k].set_title(f'State: {corresponding_model_states[i][j]}; Dim: {dim_name} ({coord})')
                    k += 1
            else:
                # plot
                ax[k].scatter(df.index, 7*df.values, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
                ax[k].fill_between(out['date'], 7*out[corresponding_model_states[i][j]].quantile(dim='draws', q=0.05/2),
                            7*out[corresponding_model_states[i][j]].quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
                ax[k].fill_between(out['date'], 7*out[corresponding_model_states[i][j]].quantile(dim='draws', q=0.50/2),
                            7*out[corresponding_model_states[i][j]].quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)
                ax[k].set_title(f'State: {corresponding_model_states[i][j]}')
                k += 1
            
            # reset output
            out = out_copy

        fig.suptitle(f'{season}')
        plt.tight_layout()
        # check if samples folder exists, if not, make it
        if not os.path.exists(path+'fit/'):
            os.makedirs(path+'fit/')
        plt.savefig(path+'fit/'+str(identifier)+'_FIT-'+f'{season}_'+run_date+'.pdf')
        plt.close()


################
## Traceplots ##
################

def traceplot(samples_xr, pars_model_shapes, hyperpars_shapes, path, identifier, run_date):
    """
    Saves traceplots for hyperparameters, as well as the model's parameters for every season
    """

    # get seasons
    seasons = list(samples_xr.coords['season'].values)

    # compute number of element-expanded parameters
    n_pars_model = sum([v[0] for v in pars_model_shapes.values()])
    n_hyperpars = sum([v[0] for v in hyperpars_shapes.values()])

    # pars_model
    ## loop over seasons
    for season in seasons:
        ## make figures
        _,axes=plt.subplots(nrows=n_pars_model, ncols=2, figsize=(8.3, 11.7/6*n_pars_model), width_ratios=[2.5,1])
        i=0
        for par_name, par_shape in pars_model_shapes.items():
            ## extract data
            s = samples_xr[par_name].sel({'season': season})
            ## build plot
            if par_shape[0] == 1:
                ax = axes[i,:]
                # traces
                ax[0].plot(s.values, color='black', alpha=0.05)
                ax[0].plot(np.median(s.values,axis=1), color='red', linewidth=1)
                ax[0].set_xlim(0, len(s.coords['iteration']))
                ax[0].set_ylabel(par_name, fontsize=9)
                # marginal distribution
                d = np.random.choice(s.values.flatten(), 5000)
                ax[1].hist(d, color='black', alpha=0.6, density=True)
                ax[1].axvline(np.median(d), color='red', linestyle='--')
            else:
                for j in range(par_shape[0]):
                    ax=axes[i+j,:]
                    # traces
                    ax[0].plot(s.values[:,:,j], color='black', alpha=0.05)
                    ax[0].plot(np.median(s.values[:,:,j],axis=1), color='red', linewidth=1)
                    ax[0].set_ylabel(f'{par_name}_{j}', fontsize=7)
                    # marginal distribution
                    d = np.random.choice(s.values[:,:,j].flatten(), 5000)
                    ax[1].hist(d, color='black', alpha=0.6, density=True)
                    ax[1].axvline(np.median(d), color='red', linestyle='--')
            i+=par_shape[0]
            
        ax[0].set_xlabel('iteration (-)', fontsize=9)
        plt.tight_layout()
        # check if samples folder exists, if not, make it
        if not os.path.exists(path+'trace/'):
            os.makedirs(path+'trace/')
        plt.savefig(path+'trace/'+str(identifier)+'_TRACE-'+f'{season}_'+run_date+'.pdf')
        plt.close()

    # hyperpars
    _,axes=plt.subplots(nrows=n_hyperpars, ncols=2, figsize=(8.3, 11.7/6*n_hyperpars), width_ratios=[2.5,1])
    i=0
    for par_name, par_shape in hyperpars_shapes.items():
        s = samples_xr[par_name]
        if par_shape[0] == 1:
            ax = axes[i,:]
            # traces
            ax[0].plot(s.values, color='black', alpha=0.05)
            ax[0].plot(np.median(s.values,axis=1), color='red', linewidth=1)
            ax[0].set_xlim(0, len(s.coords['iteration']))
            ax[0].set_ylabel(par_name, fontsize=9)
            # marginal distribution
            d = np.random.choice(s.values.flatten(), 5000)
            ax[1].hist(d, color='black', alpha=0.6, density=True)
            ax[1].axvline(np.median(d), color='red', linestyle='--')
        else:
            for j in range(par_shape[0]):
                ax=axes[i+j,:]
                # traces
                ax[0].plot(s.values[:,:,j], color='black', alpha=0.05)
                ax[0].plot(np.median(s.values[:,:,j],axis=1), color='red', linewidth=1)
                ax[0].set_ylabel(f'{par_name}_{j}', fontsize=7)
                # marginal distribution
                d = np.random.choice(s.values[:,:,j].flatten(), 5000)
                ax[1].hist(d, color='black', alpha=0.6, density=True)
                ax[1].axvline(np.median(d), color='red', linestyle='--')
        i+=par_shape[0]
        
    ax[0].set_xlabel('iteration (-)', fontsize=9)
    plt.tight_layout()
    plt.savefig(path+'trace/'+str(identifier)+'_TRACE-hyperdist_'+run_date+'.pdf')
    plt.close()

    pass