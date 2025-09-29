"""
This script handles the initialisation of the model, the formatting of the data and other miscellaneous functions
"""

__author__      = "T.W. Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group (JHUBSPH) & Bento Lab (Cornell CVM). All Rights Reserved."

import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from hierarchSIR.model import imsSIR

# all paths defined relative to this file
abs_dir = os.path.dirname(__file__)

##########################
## Model initialisation ##
##########################

def initialise_model(strains=False, fips_state=1):
    """
    A function to intialise the hierarchSIR model

    input
    -----

    - strains: bool
        - do we want a strain-stratified model?

    fips_state: int
        - '1': Alabama
    """

    # restrict input (for now)
    assert strains==1, f"only valid option for strains is 1; got {strains}"

    # parameters
    parameters = {
        # initial condition function
        'f_I': np.array(strains * [1e-4,]),
        'f_R': np.array(strains * [0.35,]),
        # SIR parameters
        'beta': np.array(strains *[0.5,]),
        'gamma': np.array(strains * [1/3.5,]),
        # modifiers
        'delta_beta_temporal': np.array([1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5])-1,
        'modifier_length': 15,
        'sigma': 2.5,
        # observation parameters
        'rho_i': 0.025,
        'rho_h': np.array(strains*[0.025,]),
        'T_h': 2.0,
        }
    
    # get inhabitants
    population = np.ones(strains) * get_demography(fips_state)

    # initialise initial condition function
    ICF = initial_condition_function(population)

    return imsSIR(parameters, ICF, strains)


class initial_condition_function():

    def __init__(self, population):
        """
        Set up the model's initial condition function

        input
        -----

        - population: int
            - number of individuals in the modeled population.
        """
        self.population = population 
        pass

    def __call__(self, f_I, f_R):
        """
        A function generating the model's initial condition; direct estimation recovered population
        
        input
        -----

        - population: int
            - Number of inhabitants in modeled US state

        - f_I: float
            - Fraction of the population initially infected
        
        - f_R: float
            - Fraction of the population initially immune

        output
        ------

        - initial_condition: dict
            - Keys: 'S0', 'I0', 'R0'. Values: int.
        """

        # construct initial condition
        return {'S0':  (1 - f_I - f_R) * self.population,
                'I0': f_I * self.population,   
                'R0': f_R * self.population,
                }

def get_demography(fips_state: int) -> int:
    """
    A function retrieving the total population of a US state

    input
    -----

    - state_fips: int
        - US state FIPS code

    output
    ------

    - population: int
        - population size
    """

    # load demography
    demography = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/demography/demography.csv'))

    return int(demography[demography['fips_state'] == fips_state]['population'].values[0])


################################
## Data and output formatting ##
################################


def extract_timestamp(fname, pattern):
    match = pattern.search(fname.name)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")
    return None

def get_latest_NHSN_HRD_influenza_data(startdate: datetime, enddate: datetime, fips_state: int, preliminary: bool) -> pd.Series:
    """
    Get the most recent NHSN HRD influenza dataset

    input
    -----

    - startdate: str/datetime
        - start of dataset
    
    - enddate: str/datetime
        - end of dataset

    - fips_state: int
        - 2 digit fips code of US state

    output
    ------

    - data: pd.DataFrame
        - index: 'date' [datetime], columns: 'H_inc'
    """

    # find most recent file
    if preliminary:
        data_folder = Path(os.path.join(abs_dir, f'../../data/interim/cases/NHSN-HRD_archive/preliminary/'))    # folder with data files
    else:
        data_folder = Path(os.path.join(abs_dir, f'../../data/interim/cases/NHSN-HRD_archive/consolidated/'))
    pattern = re.compile(r"gathered-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")                                     # regex to capture gathered timestamp
    files_with_time = [(f, extract_timestamp(f, pattern)) for f in data_folder.glob("*.parquet.gzip")]          # collect files and their timestamps
    files_with_time = [(f, t) for f, t in files_with_time if t is not None]
    latest_file, _ = max(files_with_time, key=lambda x: x[1])                                                   # get the latest file

    # print diagnostics
    #print("Most recent file:", latest_file)

    # get data
    data = pd.read_parquet(latest_file)

    # convert date column to datetime and fips_state to int
    data['date'] = pd.to_datetime(data['date'], format='ISO8601')
    data['fips_state'] = data['fips_state'].astype(int)

    # slice out relevant daterange
    data = data[((data['date'] > startdate) & (data['date'] < enddate) & (data['fips_state'] == fips_state))]

    # slice out variables of interest
    data = data[['date', 'influenza admissions']]

    # rename 'influenza admissions' to match model state name 'H_inc'
    data = data.rename(columns={'influenza admissions': 'H_inc'})  

    # set index as date and make series
    data = data.set_index('date').squeeze().sort_index()

    return data  


from pySODM.optimization.objective_functions import ll_poisson
def make_data_pySODM_compatible(start_date: datetime, end_date: datetime, fips_state: int, preliminary: bool): 
    """
    A function formatting the NHSN HRD Influenza data in pySODM format
    This involves dividing the data (which is weekly incidence) by 7 to approximate a daily incidence

    
    input:
    ------

    - start_date: datetime
        - desired startdate of data

    - end_date: datetime
        - desired enddate of data

    - fips_state: int
        - 2 digit fips code of US state

    - preliminary: bool
        - use preliminary vs. consolidated NHSN HRD data

    output:
    -------

    - data: list containing pd.DataFrame
        - contains datasets the model should be calibrated to.
    
    - states: list containing str
        - contains names of model states that should be matched to the datasets in `data`.
        - length: `len(data)`
    
    - log_likelihood_fnc: list containing log likelihood function
        - pySODM.optimization.objective_functions.ll_poisson
        - length: `len(data)`
    
    - log_likelihood_fnc_args: list containing empty lists
        - length: `len(data)`
    """

    # pySODM llp data arguments
    states = ['H_inc', ]
    log_likelihood_fnc = len(states) * [ll_poisson,]
    log_likelihood_fnc_args = len(states) * [[],]
    # pySODM data
    df = get_latest_NHSN_HRD_influenza_data(start_date, end_date, fips_state, preliminary)/7
    data = [df.dropna(), ]

    return data, states, log_likelihood_fnc, log_likelihood_fnc_args


def simout_to_hubverse(simout: xr.Dataset,
                        location: int,
                        reference_date: datetime,
                        target: str,
                        model_state: str,
                        path: str=None,
                        quantiles: bool=False) -> pd.DataFrame:
    """
    Convert simulation result to Hubverse format

    Parameters
    ----------
    - simout: xr.Dataset
        - simulation output (pySODM-compatible) . must contain `model_state`.

    - location: int
        - state FIPS code.

    - reference_date: datetime
        - when using data until a Saturday `x` to calibrate the model, `reference_date` is the date of the next saturday `x+1`.

    - target: str
        - simulation target, typically 'wk inc flu hosp'.

    - path: str
        - path to save result in. if no path provided, does not save result.

    - quantiles: str
        - save quantiles instead of individual trajectories.

    Returns
    -------

    - hubverse_df: pd.Dataframe
        - forecast in hubverse format
        - contains the total incidence in the 'value' column
        - contains the incidence per strain in the 'strain_0', 'strain_1', etc. columns

    Reference
    ---------

    https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-output/README.md#Forecast-file-format
    """

    # deduce information from simout
    location = [location,]
    output_type_id = simout.coords['draws'].values if not quantiles else [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
    strain_names = [f'strain_{name}' for name in simout[model_state].coords['strain'].values]
    # fixed metadata
    horizon = range(-1,4)
    output_type = 'samples' if not quantiles else 'quantile'
    # derived metadata
    target_end_date = [reference_date + timedelta(weeks=h) for h in horizon]

    # pre-allocate dataframe
    idx = pd.MultiIndex.from_product([[reference_date,], [target,], horizon, location, [output_type,], output_type_id],
                                        names=['reference_date', 'target', 'horizon', 'location', 'output_type', 'output_type_id'])
    df = pd.DataFrame(index=idx, columns=strain_names+['value',])
    # attach target end date
    df = df.reset_index()
    df['target_end_date'] = df.apply(lambda row: row['reference_date'] + timedelta(weeks=row['horizon']), axis=1)

    # fill in dataframe
    for loc in location:
        if not quantiles:
            for draw in output_type_id:
                # incidence per strain
                df.loc[((df['output_type_id'] == draw) & (df['location'] == loc)), strain_names] = \
                    7*simout[model_state].sel({'draws': draw}).interp(date=target_end_date).values
                # total incidence
                df.loc[((df['output_type_id'] == draw) & (df['location'] == loc)), 'value'] = \
                    7*simout[model_state].sum(dim='strain').sel({'draws': draw}).interp(date=target_end_date).values

        else:
            for q in output_type_id:
                # incidence per strain
                df.loc[((df['output_type_id'] == q) & (df['location'] == loc)), strain_names] = \
                    7*simout[model_state].quantile(q=q, dim='draws').interp(date=target_end_date).values
                # total incidence
                df.loc[((df['output_type_id'] == q) & (df['location'] == loc)), 'value'] = \
                    7*simout[model_state].quantile(q=q, dim='draws').interp(date=target_end_date).values
    
    # save result
    if path:
        df.to_csv(path+reference_date.strftime('%Y-%m-%d')+'-JHU_Cornell'+'-hierarchSIR.csv', index=False)

    return df

def samples_to_csv(ds: xr.Dataset) -> pd.DataFrame:
    """
    A function used convert the median value of parameter across MCMC chains and iterations into a flattened csv format

    Parameters
    ----------
    - ds: xarray.Dataset
        - Average or median parameter samples
        - Typically obtained after MCMC sampling in `incremental_forecasting.py` as: `ds = samples_xr.median(dim=['chain', 'iteration'])`

    Returns
    -------

    - df: pd.DataFrame
        - Index: Original parameter name
        - Columns: Element index (starts at zero) + value
    """
    parameters = []
    elements = []
    values = []

    for var_name, da in ds.data_vars.items():
        if da.ndim == 0:
            # Scalar variable
            parameters.append(var_name)
            elements.append(0)
            values.append(float(da.item()))
        elif da.ndim == 1:
            # 1D variable
            for i, val in enumerate(da.values):
                parameters.append(var_name)
                elements.append(i)
                values.append(float(val))
        else:
            raise ValueError(f"Variable '{var_name}' has more than 1 dimension ({da.dims}); this script handles only scalars and 1D variables.")

    df = pd.DataFrame(np.stack([parameters,elements,values], axis=1), columns=['parameter', 'element', 'value'])
    df['value'] = pd.to_numeric(df['value'])

    return df

from pySODM.optimization.objective_functions import log_prior_normal, log_prior_lognormal, log_prior_uniform, log_prior_gamma, log_prior_normal
def get_priors(model_name, fips_state, hyperparameters):
    """
    A function to help prepare the pySODM-compatible priors
    """

    pars = ['rho_h', 'f_R', 'f_I', 'beta', 'delta_beta_temporal']                              # parameters to calibrate
    bounds = [(0,0.02), (0,1), (1e-9,1e-2), (0.30,0.60), (-0.50,0.50)]                      # parameter bounds
    labels = [r'$\rho_{h}$',  r'$f_{R}$', r'$f_{I}$', r'$\beta$', r'$\Delta \beta_{t}$']       # labels in output figures
    # UNINFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if not hyperparameters:
        # assign priors (R0 ~ N(1.6, 0.2); modifiers nudged to zero; all others uninformative)
        log_prior_prob_fcn = [log_prior_gamma,] + [log_prior_normal,] + [log_prior_gamma,] + 2*[log_prior_normal,]
        log_prior_prob_fcn_args = [{'a': 1, 'loc': 0, 'scale': 0.05*max(bounds[0])},
                                    {'avg':  0.4, 'stdev': 0.10},
                                    {'a': 1, 'loc': 0, 'scale': 0.1*max(bounds[4])},
                                    {'avg':  0.455, 'stdev': 0.055},
                                    {'avg':  0, 'stdev': 0.10}]
    # INFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    else:
        # load and select priors
        priors = pd.read_csv(os.path.join(abs_dir, '../../data/interim/calibration/hyperparameters.csv'))
        priors = priors.loc[((priors['model'] == model_name) & (priors['fips_state'] == fips_state)), (['hyperparameter', f'{hyperparameters}'])].set_index('hyperparameter').squeeze()
        # assign values
        log_prior_prob_fcn = 1*[log_prior_lognormal,] + 1*[log_prior_normal,] + 1*[log_prior_lognormal,] + 1*[log_prior_normal,] + 12*[log_prior_normal,] 
        log_prior_prob_fcn_args = [ 
                                {'s': priors['rho_h_s_0'], 'scale': priors['rho_h_scale_0']},                                       # rho_h
                                {'avg': priors['f_R_mu_0'], 'stdev': priors['f_R_sigma_0']},                                        # f_R
                                {'s': priors['f_I_s_0'], 'scale': priors['f_I_scale_0']},                                           # f_I
                                {'avg': priors['beta_mu_0'], 'stdev': priors['beta_sigma_0']},                                      # beta
                                {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},        # delta_beta_temporal
                                {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},        # ...
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
                                ]          
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    return pars, bounds, labels, log_prior_prob_fcn, log_prior_prob_fcn_args


#########################################################
## Transmission rate: equivalent Python implementation ##
#########################################################

from scipy.ndimage import gaussian_filter1d
def get_transmission_coefficient_timeseries(modifier_vector: np.ndarray,
                                            sigma: float=2.5) -> np.ndarray:
    """
    A function mapping the modifier_vectors between Sep 15 and May 15 and smoothing it with a gaussian filter

    input
    -----

    - modifier_vector: np.ndarray
        - 1D numpy array (time) or 2D numpy array (time x spatial unit).
        - Each entry represents a value of a knotted temporal modifier, the length of each modifier is equal to the time between Oct 15 and Apr 15 (182 days) divided by `len(modifier_vector)`.

    - sigma: float 
        - gaussian smoother's standard deviation. higher values represent more smooth trajectories but increase runtime. `None` represents no smoothing (fastest).

    output
    ------

    - smooth_temporal_modifier: np.ndarray
        - 1D array of smoothed modifiers.
    """

    # Ensure the input is at least 2D
    if modifier_vector.ndim == 1:
        modifier_vector = modifier_vector[:, np.newaxis]
    _, num_space = modifier_vector.shape

    # Define number of days between Oct 15 and Apr 15
    num_days = 182

    # Step 1: Project the input vector onto the daily time scale
    interval_size = num_days / len(modifier_vector)
    positions = (np.arange(num_days) // interval_size).astype(int)
    expanded_vector = modifier_vector[positions, :]

    # Step 2: Prepend and append 31 days of ones
    padding = np.zeros((31, num_space))
    padded_vector = np.vstack([padding, expanded_vector, padding])

    # Step 3: apply the Gaussian filter
    return np.squeeze(gaussian_filter1d(padded_vector, sigma=sigma, axis=0, mode="nearest"))

##############################
## Plot fit helper function ##
##############################

def plot_fit(simout: xr.Dataset,
             data_calibration: list,
             data_validation: list,
             states: list,
             fig_path: str,
             identifier: str,
             coordinates_data_also_in_model: list,
             aggregate_over: list,
             additional_axes_data: list,
             spatial_unit: str) -> None:
    """
    A function used to visualise the goodness of fit 

    #TODO: LIMITED TO ONE COORDINATE PER DIMENSION PER DATASET !!!

    input
    -----

    - simout: xr.Dataset
        - simulation output (pySODM-compatible) . must contain all states listed in `states`.
    
    - data_calibration: list containing pySODM-compatible pd.DataFrame
        - data model was calibrated to.
        - obtained using hierarchSIR.utils.make_data_pySODM_compatible.
        - length: `len(states)`
    
    - data_validation: list containing pySODM-compatible pd.DataFrame
        - data model was not calibrated to (validation).
        - obtained using hierarchSIR.utils.make_data_pySODM_compatible.
        - length: `len(states)`

    - states: list containing str
        - names of model states that were matched with data in `data_calibration`.
    
    - fig_path: str
        - path where figure should be stored, relative to path of file this function is called from.
    
    - identifier: str
        - an ID used to name the output figure.
    
    - coordinates_data_also_in_model: list
        - contains a list for every dataset. contains a list for every model dimension besides 'date'/'time', containing the coordinates present in the data and also in the model.
        - obtained from pySODM.optimization.log_posterior_probability
    
    - aggregate_over: list
        - list of length len(data). contains, per dataset, the remaining model dimensions not present in the dataset. these are then automatically summed over while calculating the log likelihood.
        - obtained from pySODM.optimization.log_posterior_probability
    
    - additional_axes_data: list
        - axes in dataset, excluding the 'time'/'date' axes.
        - obtained from pySODM.optimization.log_posterior_probability
    
    - spatial_unit: str
        - name of the spatial unit being modeled
        - placed in the title of the plot
    """

    # check if 'draws' are provided
    samples = False
    if 'draws' in simout.dims:
        samples = True
    
    # compute the amount of timeseries to visualise
    nrows = sum(1 if not coords else len(coords) for coords in coordinates_data_also_in_model)

    # generate figure
    _,ax=plt.subplots(nrows=nrows, sharex=True, figsize=(8.3, 11.7/5*nrows))

    # vectorise ax object
    if nrows==1:
        ax = [ax,]

    # save a copy to reset
    out_copy = simout

    # loop over datasets
    k=0
    for i, (df_calib, df_valid) in enumerate(zip(data_calibration, data_validation)):
        
        # aggregate data
        for dimension in simout.dims:
            if dimension in aggregate_over[i]:
                simout = simout.sum(dim=dimension)
        
        # loop over coordinates 
        if coordinates_data_also_in_model[i]:
            for coord in coordinates_data_also_in_model[i]:
                # get dimension coord is in 
                dim_name = additional_axes_data[i][0]
                coord = coord[0]
                # plot
                ax[k].scatter(df_calib.index.get_level_values('date').values, 7*df_calib.loc[slice(None), coord].values, color='black', alpha=1, linestyle='None', facecolors='None', s=30, linewidth=1)
                if not df_valid.empty:
                    ax[k].scatter(df_valid.index.get_level_values('date').values, 7*df_valid.loc[slice(None), coord].values, color='red', alpha=1, linestyle='None', facecolors='None', s=30, linewidth=1)
                
                if samples:
                    ax[k].fill_between(simout['date'], 7*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=0.05/2),
                            7*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=1-0.05/2), color='green', alpha=0.15)
                    ax[k].fill_between(simout['date'], 7*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=0.50/2),
                            7*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=1-0.50/2), color='green', alpha=0.20)
                else:
                    ax[k].plot(simout['date'], 7*simout[states[i]].sel({dim_name: coord}), color='green')
                ax[k].set_title(f'US State: {spatial_unit}; Model state: {states[i]}; Dim: {dim_name} ({coord})')
                k += 1
        else:
            # plot
            ax[k].scatter(df_calib.index, 7*df_calib.values, color='black', alpha=1, linestyle='None', facecolors='None', s=30, linewidth=1)
            if not df_valid.empty:
                ax[k].scatter(df_valid.index, 7*df_valid.values, color='red', alpha=1, linestyle='None', facecolors='None', s=30, linewidth=1)
            if samples:
                ax[k].fill_between(simout['date'], 7*simout[states[i]].quantile(dim='draws', q=0.05/2),
                            7*simout[states[i]].quantile(dim='draws', q=1-0.05/2), color='green', alpha=0.15)
                ax[k].fill_between(simout['date'], 7*simout[states[i]].quantile(dim='draws', q=0.50/2),
                            7*simout[states[i]].quantile(dim='draws', q=1-0.50/2), color='green', alpha=0.20)
            else:
                ax[k].plot(simout['date'], 7*simout[states[i]], color='green')
            ax[k].set_title(f'US State: {spatial_unit}; Model state: {states[i]}')
            k += 1
        
        # reset output
        simout = out_copy

    plt.tight_layout()
    plt.savefig(fig_path+f'{identifier}-FIT.pdf')
    plt.close()

# helper function
def str_to_bool(value):
    """Convert string arguments to boolean (for SLURM environment variables)."""
    return value.lower() in ["true", "1", "yes"]


