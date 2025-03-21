"""
This script...
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import math
import random
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import expon, beta, norm, gamma
from pySODM.optimization.utils import list_to_dict, add_poisson_noise
from pySODM.optimization.objective_functions import ll_poisson, validate_calibrated_parameters, expand_bounds

###########################################
## Define posterior probability function ##
###########################################

def log_posterior_probability(theta, model, datasets, pars_model_names, pars_model_bounds, hyperpars_shapes, states_model, states_data):
    """
    Computes the log posterior probability

    input
    -----

    output
    ------

    """
    # pre-allocate lpp
    lpp = 0
    
    # derive the number of model parameters and all their shapes
    parameters_sizes, parameters_shapes = validate_calibrated_parameters(pars_model_names, model.parameters)
    n_pars = sum([v[0] for v in parameters_shapes.values()])

    # split the hyperparameters from the season's parameters
    n_hyperpars = sum(value[0] for value in hyperpars_shapes.values())
    theta_hyperpars = theta[:n_hyperpars]
    theta_pars = theta[n_hyperpars:]

    # and convert the hyperparameters to a dictionary
    theta_hyperpars = list_to_dict(theta_hyperpars, hyperpars_shapes, retain_floats=True)

    # compute a weights matrix
    ## pre-allocate
    w = np.ones([len(datasets), len(states_data)])
    ## weigh by inverse maximum in timeseries
    for i, data in enumerate(datasets):
        for j, state in enumerate(states_data):
            w[i,j] = 1/max(data[state].values)
    ## normalise back to one
    w /= np.mean(w)

    # loop over the seasons
    for i, data in enumerate(datasets):

        # get this season's parameters for the model
        theta_season = theta_pars[i*n_pars:(i+1)*n_pars]

        # expand the model parameters bounds (beta_temporal is 1D)
        pars_model_bounds = expand_bounds(parameters_sizes, pars_model_bounds)
        
        # Restrict this season's parameters for the model to user-provided bounds --> going outside can crash a model!
        for k,theta in enumerate(theta_season):
            if theta > pars_model_bounds[k][1]:
                theta_season[k] = pars_model_bounds[k][1]
                lpp += - np.inf
            elif theta < pars_model_bounds[k][0]:
                theta_season[k] = pars_model_bounds[k][0]
                lpp += - np.inf

        # convert to a dictionary for my ease
        theta_season = list_to_dict(theta_season, parameters_shapes, retain_floats=True)

        # compute priors of the season's parameters using the hyperparameters
        lpp += gamma.logpdf(theta_season['rho_i'], loc=0, a=theta_hyperpars['rho_i_a'], scale=theta_hyperpars['rho_i_scale'])       # rho_i
        lpp += expon.logpdf(theta_season['T_h'], scale=theta_hyperpars['T_h_rate'])                                                 # T_h
        lpp += gamma.logpdf(theta_season['rho_h'], loc=0, a=theta_hyperpars['rho_h_a'], scale=theta_hyperpars['rho_h_scale'])       # rho_h
        lpp += norm.logpdf(theta_season['beta'], loc=theta_hyperpars['beta_mu'], scale=theta_hyperpars['beta_sigma'])               # beta
        lpp += beta.logpdf(theta_season['f_R'], a=theta_hyperpars['f_R_a'], b=theta_hyperpars['f_R_b'])                             # f_R
        lpp += gamma.logpdf(theta_season['f_I'], a=theta_hyperpars['f_I_a'], loc=0, scale=theta_hyperpars['f_I_scale'])             # f_I
        lpp += np.sum(norm.logpdf(theta_season['delta_beta_temporal'], loc=theta_hyperpars['delta_beta_temporal_mu'], scale=theta_hyperpars['delta_beta_temporal_sigma']))

        # negative arguments in hyperparameters lead to a nan lpp --> redact to -np.inf and move on
        if math.isnan(lpp):
            return -np.inf

        # nor are negative betas
        if theta_hyperpars['beta_mu'] <= 0:
            return -np.inf
        
        # or huge delta_beta_temporal_mu/sigma
        if ((any(((x < -0.5) | (x > 0.5)) for x in theta_hyperpars['delta_beta_temporal_mu'])) | (any(((x < 0) | (x > 0.50)) for x in theta_hyperpars['delta_beta_temporal_sigma']))):
            return -np.inf

        # Assign model parameters (make sure they're vectors)
        model.parameters.update(theta_season)
        for par in ['rho_i', 'T_h', 'rho_h', 'beta', 'f_R', 'f_I']:
            model.parameters[par] = np.array([theta_season[par],])

        # run the forward simulation
        simout = model.sim(min(data.index), max(data.index))

        # compute the likelihood
        for j, (state_model, state_data) in enumerate(zip(states_model, states_data)):
            x = data[state_data].values
            y = simout[state_model].sum(dim='strain').interp({'date': data.index}, method='linear').values
            # check model output for nans
            if np.isnan(y).any():
                raise ValueError(f"simulation output contains nan, most likely due to numerical unstability. try using more conservative bounds.")
            lpp += w[i,j] * ll_poisson(x, y)

    return lpp

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

def hyperdistributions(samples_xr, path_filename, pars_model_shapes, bounds, N):

    # get the element-expanded number of parameters and the parameter's names
    pars_model_names = pars_model_shapes.keys()

    # make figure
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8.3,11.7/5*4))

    for _, (ax, par_name, bound) in enumerate(zip(axes.flatten(), pars_model_names, bounds)):
        
        # define x based on plausible range
        x = np.linspace(start=bound[0],stop=bound[1],num=100)

        ## EXPONENTIAL
        if par_name == 'T_h':
            # draw a random chain and iteration
            for _ in range(N):
                i = random.randint(0, len(samples_xr.coords['iteration'])-1)
                j = random.randint(0, len(samples_xr.coords['chain'])-1)
                ax.plot(x, expon.pdf(x, scale=samples_xr['T_h_rate'].sel({'iteration': i, 'chain': j}).values), alpha=0.05, color='black')        
            # draw mean
            m = samples_xr['T_h_rate'].median(dim=['iteration', 'chain']).values
            ax.plot(x, expon.pdf(x, scale=m), color='red', linestyle='--')
            # add parameter box
            ax.text(0.05, 0.95, f"scale={m:.1f}", transform=ax.transAxes, fontsize=7,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
            ax.set_ylabel('$T_h$')
        ## GAMMA
        elif par_name in ['rho_i', 'rho_h', 'f_I']:
            a_name = f'{par_name}_a'
            scale_name = f'{par_name}_scale'
            # draw a random chain and iteration
            for _ in range(N):
                i = random.randint(0, len(samples_xr.coords['iteration'])-1)
                j = random.randint(0, len(samples_xr.coords['chain'])-1)
                ax.plot(x, gamma.pdf(x, a=samples_xr[a_name].sel({'iteration': i, 'chain': j}).values, scale=samples_xr[scale_name].sel({'iteration': i, 'chain': j}).values), alpha=0.05, color='black')        
            # draw mean
            ax.plot(x, gamma.pdf(x, a=samples_xr[a_name].median(dim=['iteration', 'chain']).values, scale=samples_xr[scale_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
            # add parameter box
            ax.text(0.05, 0.95, f"a={samples_xr[a_name].median(dim=['iteration', 'chain']).values:.1e}, scale={samples_xr[scale_name].median(dim=['iteration', 'chain']).values:.1e}", transform=ax.transAxes, fontsize=7,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
            ax.set_ylabel(par_name)
        ## NORMAL
        elif par_name in ['beta',]:
            mu_name = f'{par_name}_mu'
            sigma_name = f'{par_name}_sigma'
            # draw a random chain and iteration
            for _ in range(N):
                i = random.randint(0, len(samples_xr.coords['iteration'])-1)
                j = random.randint(0, len(samples_xr.coords['chain'])-1)
                ax.plot(x, norm.pdf(x, loc=samples_xr[mu_name].sel({'iteration': i, 'chain': j}).values, scale=samples_xr[sigma_name].sel({'iteration': i, 'chain': j}).values), alpha=0.05, color='black')        
            # draw mean
            ax.plot(x, norm.pdf(x, loc=samples_xr[mu_name].median(dim=['iteration', 'chain']).values, scale=samples_xr[sigma_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
            # add parameter box
            ax.text(0.05, 0.95, f"avg={samples_xr[mu_name].median(dim=['iteration', 'chain']).values:.1e}, stdev={samples_xr[sigma_name].median(dim=['iteration', 'chain']).values:.1e}", transform=ax.transAxes, fontsize=7,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
            ax.set_ylabel(par_name)
        ## BETA
        elif par_name in ['f_R',]:
            a_name = f'{par_name}_a'
            b_name = f'{par_name}_b'
            # draw a random chain and iteration
            for _ in range(N):
                i = random.randint(0, len(samples_xr.coords['iteration'])-1)
                j = random.randint(0, len(samples_xr.coords['chain'])-1)
                ax.plot(x, beta.pdf(x, a=samples_xr[a_name].sel({'iteration': i, 'chain': j}).values, b=samples_xr[b_name].sel({'iteration': i, 'chain': j}).values), alpha=0.05, color='black')        
            # draw mean
            ax.plot(x, beta.pdf(x, a=samples_xr[a_name].median(dim=['iteration', 'chain']).values, b=samples_xr[b_name].median(dim=['iteration', 'chain']).values), color='red', linestyle='--')
            # add parameter box
            ax.text(0.05, 0.95, f"a={samples_xr[a_name].median(dim=['iteration', 'chain']).values:.1f}, b={samples_xr[b_name].median(dim=['iteration', 'chain']).values:.1f}", transform=ax.transAxes, fontsize=7,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
            ax.set_ylabel(par_name)
        ## TEMPORAL BETAS
        elif par_name == 'delta_beta_temporal':
            pass
            # ### get transmission rate function
            # from influenza_USA.NC_forecasts.TDPF import transmission_rate_function
            # f = transmission_rate_function(sigma=2.5)
            # x = pd.date_range(start=datetime(2020,10,21), end=datetime(2021,4,10), freq='2D').tolist()
            # ### compute modifier tranjectory of every season and plot
            # for i, season in enumerate(samples_xr.coords['season']):
            #     y = []
            #     for d in x:
            #         y.append(f(d, {}, 1, samples_xr['delta_beta_temporal'].median(dim=['iteration', 'chain']).sel(season=season).values))
            #     ax.plot(x, np.squeeze(np.array(y)), color='black', linewidth=0.5, alpha=0.2)
            # ### visualise hyperdistribution
            # ll=[]
            # y=[]
            # ul=[]
            # for d in x:
            #     ll.append(f(d, {}, 1, samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values - samples_xr['delta_beta_temporal_sigma'].median(dim=['iteration', 'chain']).values))
            #     y.append(f(d, {}, 1, samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values))
            #     ul.append(f(d, {}, 1, samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values + samples_xr['delta_beta_temporal_sigma'].median(dim=['iteration', 'chain']).values))
            # ax.plot(x, np.squeeze(np.array(y)), color='red', alpha=0.8)
            # ax.fill_between(x, np.squeeze(np.array(ll)), np.squeeze(np.array(ul)), color='red', alpha=0.1)
            # add parameter box
            #ax.text(0.02, 0.97, f"avg={list(np.round(samples_xr['delta_beta_temporal_mu'].median(dim=['iteration', 'chain']).values,2))}\nstdev={list(np.round(samples_xr['delta_beta_temporal_sigma'].median(dim=['iteration', 'chain']).values,2))}", transform=ax.transAxes, fontsize=5,
            #    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
            #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            #ax.set_ylabel(r'$\Delta \beta_{t}$')
            #ax.set_ylim([0.7, 1.3])

    fig.delaxes(axes[3,1])
    plt.tight_layout()
    plt.savefig(path_filename)
    plt.close()

    pass


#####################
## Goodness-of-fit ##
#####################

import random
def draw_function(parameters, samples_xr, season, pars_model_names):
    """
    A compatible draw function
    """

    # get a random iteration and markov chain
    i = random.randint(0, len(samples_xr.coords['iteration'])-1)
    j = random.randint(0, len(samples_xr.coords['chain'])-1)
    # assign parameters
    for var in pars_model_names:
        if var != 'delta_beta_temporal':
            parameters[var] = np.array([samples_xr[var].sel({'iteration': i, 'chain': j, 'season': season}).values],)
        else:
            parameters[var] = samples_xr[var].sel({'iteration': i, 'chain': j, 'season': season}).values
    return parameters

def plot_fit(model, datasets, samples_xr, pars_model_names, path, identifier, run_date):
    """
    Visualises the goodness of fit for every season
    """

    # simulate model for every season
    simout=[]
    for season, data in zip(list(samples_xr.coords['season'].values), datasets):
        simout.append(add_poisson_noise(model.sim(min(data.index), max(data.index), N=100,
                                        draw_function=draw_function, draw_function_kwargs={'samples_xr': samples_xr, 'season': season, 'pars_model_names': pars_model_names})+0.01
                                        ))

    # visualise outcome
    for season, data, out in zip(list(samples_xr.coords['season'].values), datasets, simout):
        
        fig,ax=plt.subplots(nrows=2, figsize=(8.3, 11.7/5*2))
        # hosp
        ax[0].scatter(data.index, 7*data['H_inc'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[0].fill_between(out['date'], 7*out['H_inc'].sum(dim=['strain',]).quantile(dim='draws', q=0.05/2),
                            7*out['H_inc'].sum(dim='strain').quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
        ax[0].fill_between(out['date'], 7*out['H_inc'].sum(dim='strain').quantile(dim='draws', q=0.50/2),
                            7*out['H_inc'].sum(dim='strain').quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)
        ax[0].set_title(f'Hospitalisations')
        ax[0].set_ylabel('Weekly hospital inc. (-)')
        # ILI
        ax[1].scatter(data.index, 7*data['I_inc'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[1].fill_between(out['date'], 7*out['I_inc'].sum(dim='strain').quantile(dim='draws', q=0.05/2),
                            7*out['I_inc'].sum(dim='strain').quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
        ax[1].fill_between(out['date'], 7*out['I_inc'].sum(dim='strain').quantile(dim='draws', q=0.50/2),
                            7*out['I_inc'].sum(dim='strain').quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)   
        ax[1].set_title(f'Influenza-like illness')
        ax[1].set_ylabel('Weekly ILI inc. (-)')
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