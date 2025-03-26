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
from scipy.stats import expon, beta, norm, gamma
from pySODM.optimization.utils import list_to_dict, add_poisson_noise
from pySODM.optimization.objective_functions import ll_poisson, validate_calibrated_parameters, expand_bounds, validate_dataset, create_fake_xarray_output, compare_data_model_coordinates

###########################################
## Define posterior probability function ##
###########################################

class log_posterior_probability():

    def __init__(self, model, par_names, par_bounds, par_hyperdistributions, datasets):

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
            elif hyperdist == 'normal':
                hyperpar_shapes[f'{name}_mu'] = shape
                hyperpar_shapes[f'{name}_sigma'] = shape
            elif hyperdist == 'beta':
                hyperpar_shapes[f'{name}_a'] = shape
                hyperpar_shapes[f'{name}_b'] = shape
            else:
                ValueError(f"'{hyperdist}' is not a valid hyperdistribution.")
        self.hyperpar_shapes = hyperpar_shapes
        self.n_hyperpars = sum(value[0] for value in self.hyperpar_shapes.values())

        # get additional axes beside time axis in dataset and model states we want to match
        self.corresponding_model_states = []
        self.additional_axes_data = []
        for data in datasets:
            states, addaxis = validate_dataset(data)
            self.corresponding_model_states.append(states)
            self.additional_axes_data.append(addaxis)

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
        for data, states, addaxdata in zip(datasets, self.corresponding_model_states, self.additional_axes_data):
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
        self.par_bounds = par_bounds
        self.datasets = datasets

        pass

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
        
        # pre-allocate lpp
        lpp = 0

        # split the hyperparameters from the season's parameters
        theta_hyperpars = theta[:self.n_hyperpars]
        theta_pars = theta[self.n_hyperpars:]

        # and convert the hyperparameters to a dictionary
        theta_hyperpars = list_to_dict(theta_hyperpars, self.hyperpar_shapes, retain_floats=True)

        # loop over the seasons
        for i, data in enumerate(self.datasets):

            # get this season's parameters for the model
            theta_season = theta_pars[i*self.n_pars:(i+1)*self.n_pars]

            # expand the model parameters bounds (beta_temporal is 1D)
            pars_model_bounds = expand_bounds(self.par_sizes, self.par_bounds)
            
            # Restrict this season's parameters for the model to user-provided bounds --> going outside can crash a model!
            for k,theta in enumerate(theta_season):
                if theta > pars_model_bounds[k][1]:
                    theta_season[k] = pars_model_bounds[k][1]
                    lpp += - np.inf
                elif theta < pars_model_bounds[k][0]:
                    theta_season[k] = pars_model_bounds[k][0]
                    lpp += - np.inf

            # convert to a dictionary for ease
            theta_season = list_to_dict(theta_season, self.par_shapes, retain_floats=True)

            # compute priors
            for pars_model_name, pars_model_hyperdistribution in zip(self.par_names, self.par_hyperdistributions):
                if pars_model_hyperdistribution == 'gamma':
                    ### construct hyperpars names
                    a_name = f'{pars_model_name}_a'
                    scale_name = f'{pars_model_name}_scale'
                    ### compute lpp
                    lpp += np.sum(gamma.logpdf(theta_season[pars_model_name], loc=0, a=theta_hyperpars[a_name], scale=theta_hyperpars[scale_name]))
                elif pars_model_hyperdistribution == 'expon':
                    ### construct hyperpars names
                    scale_name = f'{pars_model_name}_scale'
                    ### compute lpp
                    lpp += np.sum(expon.logpdf(theta_season[pars_model_name], scale=theta_hyperpars[scale_name]))
                elif pars_model_hyperdistribution == 'normal':
                    ### construct hyperpars names
                    mu_name = f'{pars_model_name}_mu'
                    sigma_name = f'{pars_model_name}_sigma'
                    ### compute lpp
                    lpp += np.sum(norm.logpdf(theta_season[pars_model_name], loc=theta_hyperpars[mu_name], scale=theta_hyperpars[sigma_name]))   
                elif pars_model_hyperdistribution == 'beta':
                    ### construct hyperpars names
                    a_name = f'{pars_model_name}_a'
                    b_name = f'{pars_model_name}_b'
                    ### compute lpp
                    lpp += np.sum(beta.logpdf(theta_season[pars_model_name], a=theta_hyperpars[a_name], b=theta_hyperpars[b_name]))       

            # negative arguments in hyperparameters lead to a nan lpp --> redact to -np.inf and move on
            if math.isnan(lpp):
                return -np.inf
            # nor are negative betas
            #if (any(x <= 0) for x in theta_hyperpars['beta_mu']):
            #    return -np.inf
            # or huge delta_beta_temporal_mu/sigma
            if ((any(((x < -1) | (x > 1)) for x in theta_hyperpars['delta_beta_temporal_mu'])) | (any(((x < 0) | (x > 1)) for x in theta_hyperpars['delta_beta_temporal_sigma']))):
                return -np.inf

            # Assign model parameters
            self.model.parameters.update(theta_season)
            # But make sure they're vectors (if using one strain)
            for par in self.par_names:
                if ((par != 'delta_beta_temporal') & (self.model.parameter_shapes[par] == (1,))):
                    self.model.parameters[par] = np.array([theta_season[par],])

            # run the forward simulation
            simout = self.model.sim(self.simtimes[i])

            # loop over states
            for j, (state_model, df) in enumerate(zip(self.corresponding_model_states[i], data)):
                # reset copy
                out_copy = simout
                # aggregate over unwanted dimensions
                for dimension in simout.dims:
                    if dimension in self.aggregate_over[i][j]:
                        out_copy = out_copy.sum(dim=dimension)
                # match data and model
                if not self.additional_axes_data[i][j]:
                    # get timeseries
                    x = df.squeeze().values
                    y = out_copy[state_model].sel({'date': df.index.get_level_values('date').unique().values}).values
                    # check if shapes are consistent
                    if x.shape != y.shape:
                        raise Exception(f"shape of model prediction {y.shape} and data {x.shape} are not identical.")
                    # check for nan in model output
                    if np.isnan(y).any():
                        raise ValueError(f"simulation output contains nan, most likely due to numerical unstability. try using more conservative bounds.")
                    # compute lpp
                    lpp += self.w[i,j] * ll_poisson(x, y)
                else:
                    # get timeseries
                    x = df.squeeze().values
                    y = np.squeeze(out_copy[state_model].sel({'date': df.index.get_level_values('date').unique().values}).sel({k:self.coordinates_data_also_in_model[i][j][jdx] for jdx,k in enumerate(self.additional_axes_data[i][j])}).values)
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

def hyperdistributions(samples_xr, path_filename, pars_model_shapes, pars_model_hyperdistributions, bounds, N):

    # get the element-expanded number of parameters and the parameter's names
    pars_model_names = pars_model_shapes.keys()

    # make figure
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8.3,11.7/5*4))

    for _, (ax, par_name, par_hyperdistribution, bound) in enumerate(zip(axes.flatten(), pars_model_names, pars_model_hyperdistributions, bounds)):

        # exception for `delta_beta_temporal`
        if par_name == 'delta_beta_temporal':

            ### get transmission rate function
            from hierarchSIR.utils import get_transmission_coefficient_timeseries
            ### compute modifier tranjectory of every season and plot
            for i, season in enumerate(samples_xr.coords['season']):
                y = 1 + get_transmission_coefficient_timeseries(samples_xr['delta_beta_temporal'].median(dim=['iteration', 'chain']).sel(season=season).values, sigma=2.5)
                x = pd.date_range(start=datetime(2020,9,15), end=datetime(2020,9,15) + timedelta(days=len(y)-1), freq='D').tolist()
                ax.plot(x, y, color='black', linewidth=0.5, alpha=0.2)
            ### visualise hyperdistribution
            ll= 1 + get_transmission_coefficient_timeseries(samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values - samples_xr['delta_beta_temporal_sigma'].median(dim=['iteration', 'chain']).values, sigma=2.5)
            y= 1 + get_transmission_coefficient_timeseries(samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values, sigma=2.5)
            ul=1 + get_transmission_coefficient_timeseries(samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values + samples_xr['delta_beta_temporal_sigma'].median(dim=['iteration', 'chain']).values, sigma=2.5)
            ax.plot(x, y, color='red', alpha=0.8)
            ax.fill_between(x, ll, ul, color='red', alpha=0.1)
            # add parameter box
            ax.text(0.02, 0.97, f"avg={list(np.round(samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values,2).tolist())}\nstdev={list(np.round(samples_xr['delta_beta_temporal_sigma'].median(dim=['iteration', 'chain']).values,2).tolist())}", transform=ax.transAxes, fontsize=5,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.set_ylabel(r'$\Delta \beta_{t}$')
            ax.set_xlim([datetime(2020,10,21), datetime(2021, 4, 12)])
            ax.set_ylim([0.7, 1.3])

        else:

            # define x based on plausible range
            x = np.linspace(start=bound[0],stop=bound[1],num=100)

            # loop over the number of iterations
            for k in range(N):

                ## draw a random sample
                i = random.randint(0, len(samples_xr.coords['iteration'])-1)
                j = random.randint(0, len(samples_xr.coords['chain'])-1)

                ## if not `delta_beta_temporal`: select the right distribution
                if par_hyperdistribution == 'gamma':
                    ### construct hyperpars names
                    a_name = f'{par_name}_a'
                    scale_name = f'{par_name}_scale'
                    ### add sample to plot
                    ax.plot(x, gamma.pdf(x, a=samples_xr[a_name].sel({'iteration': i, 'chain': j}).values, scale=samples_xr[scale_name].sel({'iteration': i, 'chain': j}).values), alpha=0.05, color='black')
                    ### add mean
                    if k == N-1:
                        #### mean
                        ax.plot(x, gamma.pdf(x, a=samples_xr[a_name].median(dim=['iteration', 'chain']).values, scale=samples_xr[scale_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                        #### textbox
                        ax.text(0.05, 0.95, f"a={samples_xr[a_name].median(dim=['iteration', 'chain']).values:.1e}, scale={samples_xr[scale_name].median(dim=['iteration', 'chain']).values:.1e}", transform=ax.transAxes, fontsize=7,
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                        ax.set_ylabel(par_name)
                elif par_hyperdistribution == 'expon':
                    ### construct hyperpars names
                    scale_name = f'{par_name}_scale'
                    ### add sample to plot
                    ax.plot(x, expon.pdf(x, scale=samples_xr[scale_name].sel({'iteration': i, 'chain': j}).values), alpha=0.05, color='black')
                    ### add mean
                    if k == N-1:
                        #### mean
                        ax.plot(x, expon.pdf(x, scale=samples_xr[scale_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                        #### textbox
                        ax.text(0.05, 0.95, f"scale={samples_xr[scale_name].median(dim=['iteration', 'chain']).values:.1f}", transform=ax.transAxes, fontsize=7,
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                        ax.set_ylabel('$T_h$')
                elif par_hyperdistribution == 'normal':
                    ### construct hyperpars names
                    mu_name = f'{par_name}_mu'
                    sigma_name = f'{par_name}_sigma'
                    ### add sample to plot
                    ax.plot(x, norm.pdf(x, loc=samples_xr[mu_name].sel({'iteration': i, 'chain': j}).values, scale=samples_xr[sigma_name].sel({'iteration': i, 'chain': j}).values), alpha=0.05, color='black')        
                    ### add mean
                    if k == N-1:
                        #### mean
                        ax.plot(x, norm.pdf(x, loc=samples_xr[mu_name].median(dim=['iteration', 'chain']).values, scale=samples_xr[sigma_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                        #### textbox
                        ax.text(0.05, 0.95, f"avg={samples_xr[mu_name].median(dim=['iteration', 'chain']).values:.1e}, stdev={samples_xr[sigma_name].median(dim=['iteration', 'chain']).values:.1e}", transform=ax.transAxes, fontsize=7,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                        ax.set_ylabel(par_name)
                elif par_hyperdistribution == 'beta':
                    ### construct hyperpars names
                    a_name = f'{par_name}_a'
                    b_name = f'{par_name}_b'
                    ### add sample to plot
                    ax.plot(x, beta.pdf(x, a=samples_xr[a_name].sel({'iteration': i, 'chain': j}).values, b=samples_xr[b_name].sel({'iteration': i, 'chain': j}).values), alpha=0.05, color='black')        
                    ### add mean
                    if k == N-1:
                        #### mean
                        ax.plot(x, beta.pdf(x, a=samples_xr[a_name].median(dim=['iteration', 'chain']).values, b=samples_xr[b_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
                        #### textbox
                        ax.text(0.05, 0.95, f"a={samples_xr[a_name].median(dim=['iteration', 'chain']).values:.1f}, b={samples_xr[b_name].median(dim=['iteration', 'chain']).values:.1f}", transform=ax.transAxes, fontsize=7,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
                        ax.set_ylabel(par_name)

    fig.delaxes(axes[3,1])
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