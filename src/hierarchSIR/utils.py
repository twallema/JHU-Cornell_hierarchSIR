import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from hierarchSIR.model import SIR

def initialise_model(strains=False, fips_state=37):
    """
    A function to intialise the model
    """

    if strains == True:
        n_strains = 2
        # Parameters
        parameters = {
        # initial condition function
        'f_I': np.array([1e-4, 1e-6]),
        'f_R': np.array([0.35, 0.35]), 
        # SIR parameters
        'beta': [0.5, 0.5],
        'gamma': [1/3.5, 1/3.5],
        # modifiers
        'delta_beta_temporal': np.array([1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5])-1,
        'modifier_length': 15,
        'sigma': 2.5,
        # observation parameters
        'rho_i': [0.025, 0.025],
        'rho_h': [0.025, 0.025],
        'T_h': 3.5
        }
    else:
        n_strains = 1
        # Parameters
        parameters = {
        # initial condition function
        'f_I': np.array([1e-4,]),
        'f_R': np.array([0.35,]), 
        # SIR parameters
        'beta': [0.5,],
        'gamma': [1/3.5,],
        # modifiers
        'delta_beta_temporal': np.array([1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5])-1,
        'modifier_length': 15,
        'sigma': 2.5,
        # observation parameters
        'rho_i': [0.025,],
        'rho_h': [0.0025,],
        'T_h': 3.5
        }

    # get inhabitants
    population = np.ones(n_strains) * get_demography(fips_state)

    # initialise initial condition function
    ICF = initial_condition_function(population)

    return SIR(parameters, ICF, n_strains)

class initial_condition_function():

    def __init__(self, population):
        self.population = population 
        pass

    def __call__(self, f_I, f_R):
        """
        A function generating the model's initial condition.
        
        input
        -----

        population: int
            Number of inhabitants in modeled US state

        f_I: float
            Fraction of the population initially infected
        
        f_R: float
            Fraction of the population initially immune

        output
        ------

        initial_condition: dict
            Keys: 'S0', ... . Values: np.ndarray.
        """

        # construct initial condition
        return {'S0':  (1 - f_I - f_R) * self.population,
                'I0': f_I * self.population,   
                'R0': f_R * self.population,
                }

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

def get_demography(fips_state):
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
    demography = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/interim/demography/demography.csv'))

    return int(demography[demography['fips_state'] == fips_state]['population'].values[0])

def get_NC_influenza_data(startdate, enddate, season):
    """
    Get the North Carolina Influenza dataset -- containing ED visits, ED admissions and subtype information -- for a given season

    input
    -----

    - startdate: str/datetime
        - start of dataset
    
    - enddate: str/datetime
        - end of dataset

    - season: str
        - influenza season

    output
    ------

    - data: pd.DataFrame
        - index: 'date' [datetime], columns: 'H_inc', 'I_inc', 'H_inc_A', 'H_inc_B' (frequency: weekly, converted to daily)
    """

    # load raw Hospitalisation and ILI data + convert to daily incidence
    data_raw = [
        pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/raw/cases/hosp-admissions_NC_2010-2025.csv'), index_col=0, parse_dates=True)[['flu_hosp']].squeeze()/7,  # hosp
        pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/raw/cases/ED-visits_NC_2010-2025.csv'), index_col=0, parse_dates=True)[['flu_ED']].squeeze()/7               # ILI
            ]   
    # rename 
    data_raw[0] = data_raw[0].rename('H_inc')
    data_raw[1] = data_raw[1].rename('I_inc')
    # merge
    data_raw = pd.concat(data_raw, axis=1)
    # change index name
    data_raw.index.name = 'date'
    # slice right dates
    data_raw = data_raw.loc[slice(startdate,enddate)]
    # load subtype data flu A vs. flu B
    df_subtype = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/interim/cases/subtypes_NC_14-25.csv'), index_col=1, parse_dates=True)
    # load right season
    df_subtype = df_subtype[df_subtype['season']==season][['flu_A', 'flu_B']]
    # merge with the epi data
    df_merged = pd.merge(data_raw, df_subtype, how='outer', left_on='date', right_on='date')
    # assume a 50/50 ratio where no subtype data is available
    df_merged[['flu_A', 'flu_B']] = df_merged[['flu_A', 'flu_B']].fillna(1)
    # compute fraction of Flu A
    df_merged['fraction_A'] = df_merged['flu_A'] / (df_merged['flu_A'] + df_merged['flu_B']) # compute percent A
    # re-compute flu A and flu B cases
    df_merged['H_inc_A'] = df_merged['H_inc'] * df_merged['fraction_A']
    df_merged['H_inc_B'] = df_merged['H_inc'] * (1-df_merged['fraction_A'])
    # throw out rows with na
    df_merged = df_merged.dropna()
    # throw out `fraction_A`
    return df_merged[['H_inc', 'I_inc', 'H_inc_A', 'H_inc_B']].loc[slice(startdate,enddate)]


def pySODM_to_hubverse(simout: xr.Dataset,
                        location: int,
                        reference_date: datetime,
                        target: str,
                        model_state: str,
                        path: str=None,
                        quantiles: bool=False) -> pd.DataFrame:
    """
    Convert pySODM simulation result to Hubverse format

    Parameters
    ----------
    - simout: xr.Dataset
        - pySODM simulation output. must contain `model_state`.

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

    Reference
    ---------

    https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-output/README.md#Forecast-file-format
    """

    # deduce information from simout
    location = [location,]
    output_type_id = simout.coords['draws'].values if not quantiles else [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
    # fixed metadata
    horizon = range(-1,4)
    output_type = 'samples' if not quantiles else 'quantile'
    # derived metadata
    target_end_date = [reference_date + timedelta(weeks=h) for h in horizon]

    # pre-allocate dataframe
    idx = pd.MultiIndex.from_product([[reference_date,], [target,], horizon, location, [output_type,], output_type_id],
                                        names=['reference_date', 'target', 'horizon', 'location', 'output_type', 'output_type_id'])
    df = pd.DataFrame(index=idx, columns=['value'])
    # attach target end date
    df = df.reset_index()
    df['target_end_date'] = df.apply(lambda row: row['reference_date'] + timedelta(weeks=row['horizon']), axis=1)

    # fill in dataframe
    for loc in location:
        if not quantiles:
            for draw in output_type_id:
                df.loc[((df['output_type_id'] == draw) & (df['location'] == loc)), 'value'] = \
                    7*simout[model_state].sum(dim='strain').sel({'draws': draw}).interp(date=target_end_date).values
        else:
            for q in output_type_id:
                df.loc[((df['output_type_id'] == q) & (df['location'] == loc)), 'value'] = \
                    7*simout[model_state].sum(dim='strain').quantile(q=q, dim='draws').interp(date=target_end_date).values
    
    # save result
    if path:
        df.to_csv(path+reference_date.strftime('%Y-%m-%d')+'-JHU_IDD'+'-hierarchSIM.csv', index=False)

    return df

#########################################################
## Transmission rate: equivalent Python implementation ##
#########################################################

import math
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

class transmission_rate_function():

    def __init__(self, sigma):
        self.sigma = sigma
        pass

    def __call__(self, t, beta_0, delta_beta_temporal):
        """
        A function constructing a temporal transmission rate 'beta'

        input
        -----

        - t: datetime.datetime
            - time in simulation

        - beta_0: float
            - baseline transmission modifier

        - delta_beta_temporal: np.ndarray
            - multiplicative piecewise-continuous modifier of transmission rate between mid Oct - mid Apr (183 days)
            - no effect: modifier = 0
            - biweekly (15d): length 12

        output
        ------

        - beta(t): np.ndarray
            - ime-varying transmission rate
        """

        # smooth modifier
        temporal_modifiers_smooth = self.get_smooth_temporal_modifier(1+np.array(delta_beta_temporal), t, sigma=self.sigma)

        # apply modifier
        return beta_0 * temporal_modifiers_smooth


    @staticmethod
    def get_smooth_temporal_modifier(modifier_vector, simulation_date, sigma=None):
        """
        A function returning the value of a temporal modifier on `simulation_date` after smoothing with a gaussian filter

        input
        -----

        - modifier_vector: np.ndarray
            - 1D numpy array (time) or 2D numpy array (time x spatial unit).
            - Each entry represents a value of a knotted temporal modifier, the length of each modifier is equal to the number of days between Oct 15 and Apr 15 divided by `len(modifier_vector)`.

        - simulation_date: datetime
            - current simulation date

        - sigma: float or None
            - gaussian smoother's standard deviation. higher values represent more smooth trajectories but increase runtime. None represents no smoothing (fastest).

        output
        ------

        - smooth_temporal_modifier: float
            - smoothed modifier at `simulation_date`
            - 1D array of smoothed modifiers at `simulation_date`. If the input is 1D, the output will be a single-element array. If the input is 2D, the output will have one value for each spatial dimension.
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
        padding = np.ones((31, num_space))
        padded_vector = np.vstack([padding, expanded_vector, padding])

        # Step 3: Compute the number of days since the last Sept 15
        year = simulation_date.year
        # Compute the last October 1
        sept15_this_year = datetime(year, 9, 15)
        if simulation_date >= sept15_this_year:
            last_sept15 = sept15_this_year
        else:
            last_sept15 = datetime(year - 1, 9, 15)
        # Calculate the difference in days
        days_difference = (simulation_date - last_sept15).days

        # Step 4: If outside of range return 1
        if days_difference < 0 or days_difference >= padded_vector.shape[0]:
            return np.ones(num_space)  # Default value if out of range

        # Step 5: apply the Gaussian filter only within a +- 4*sigma window
        if not sigma:
            sigma = 1 # just pick something
            lower_bound = max(0, days_difference - math.ceil(4 * sigma))
            upper_bound = min(padded_vector.shape[0], days_difference + math.ceil(4 * sigma)+1)
            smoothed_subseries = padded_vector[lower_bound:upper_bound]
        else:
            lower_bound = max(0, days_difference - math.ceil(4 * sigma))
            upper_bound = min(padded_vector.shape[0], days_difference + math.ceil(4 * sigma)+1)
            smoothed_subseries = gaussian_filter1d(padded_vector[lower_bound:upper_bound], sigma=sigma, axis=0, mode="nearest")

        # Return the smoothed value at the correct relative position
        return smoothed_subseries[days_difference - lower_bound, :]