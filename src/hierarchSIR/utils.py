import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from hierarchSIR.model import SIR

##########################
## Model initialisation ##
##########################

def initialise_model(strains=False, immunity_linking=False, season=None, fips_state=37):
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
        'rho_i': [0.025,],
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
    if immunity_linking:
        if strains:
            historic_cumulative_incidence = get_NC_cumulatives_per_season()[['H_inc_A', 'H_inc_B']]
        else:
            historic_cumulative_incidence = get_NC_cumulatives_per_season()['H_inc']
        ICF = initial_condition_function(population, historic_cumulative_incidence).w_immunity_linking
    else:
        historic_cumulative_incidence = get_NC_cumulatives_per_season()['H_inc']
        ICF = initial_condition_function(population, historic_cumulative_incidence).wo_immunity_linking

    # adjust parameters dictionary
    if immunity_linking:
        del parameters['f_R']
        parameters['season'] = season
        parameters['iota_1'] = parameters['iota_2'] = parameters['iota_3'] = np.ones(n_strains) * 1e-5

    return SIR(parameters, ICF, n_strains)


class initial_condition_function():

    def __init__(self, population, historic_cumulative_incidence):
        self.population = population 
        self.historic_cumulative_incidence = historic_cumulative_incidence
        pass

    def wo_immunity_linking(self, f_I, f_R):
        """
        A function generating the model's initial condition -- no immunity linking; direct estimation recovered population
        
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
    
    def w_immunity_linking(self, f_I, iota_1, iota_2, iota_3,  season):
        """
        A function setting the model's initial condition.
        
        input
        -----

        f_I: float
            Fraction of the population initially infected
        
        iota_n: float
            Relative influence of cumulative cases n seasons ago -- used to compute the fraction of the population initially immune


        output
        ------

        initial_condition: dict
            Keys: 'S', ... . Values: np.ndarray (n_age x n_loc).
        """

        # immunity link function
        ##  get data
        if len(iota_1) > 1:
            C_min1 = self.historic_cumulative_incidence.loc[(season,-1)].values
            C_min2 = self.historic_cumulative_incidence.loc[(season,-2)].values
            C_min3 = self.historic_cumulative_incidence.loc[(season,-3)].values
        else:
            C_min1 = self.historic_cumulative_incidence.loc[(season,-1)]
            C_min2 = self.historic_cumulative_incidence.loc[(season,-2)]
            C_min3 = self.historic_cumulative_incidence.loc[(season,-3)]
        ## flatten parameters
        iota_1 = np.squeeze(iota_1)
        iota_2 = np.squeeze(iota_2)
        iota_3 = np.squeeze(iota_3)
        
        ## compute immunity (bounded linear model)
        f_R = (iota_1 * C_min1 + iota_2 * C_min2 + iota_3 * C_min3) / (1 + iota_1 * C_min1 + iota_2 * C_min2 + iota_3 * C_min3)

        return {'S0':  (1 - f_I - f_R) * self.population,
                'I0': f_I * self.population,   
                'R0': f_R * self.population,
                }

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


################################
## Data and output formatting ##
################################

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


def get_NC_cumulatives_per_season():
    """
    A function that returns, for each season, the cumulative total H_inc, I_inc, H_inc_A and H_inc_B in the season - 0, season - 1 and season - 2.
    """
    # define seasons we want output for
    seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2023-2024', '2024-2025']

    # loop over them
    seasons_collect = []
    for season in seasons:
        # get the season start
        season_start = int(season[0:4])
        # go back two seasons
        horizons_collect = []
        for i in [0, -1, -2, -3]:
            # get the data
            data = get_NC_influenza_data(datetime(season_start+i,10,1), datetime(season_start+1+i,5,1), f'{season_start+i}-{season_start+1+i}')*7
            # calculate cumulative totals
            column_sums = {
                "horizon": i,
                "H_inc": data["H_inc"].sum(),
                "I_inc": data["I_inc"].sum(),
                "H_inc_A": data["H_inc_A"].sum(),
                 "H_inc_B": data["H_inc_B"].sum(),
            }
            # create the DataFrame
            horizons_collect.append(pd.DataFrame([column_sums]))
        # concatenate data
        data = pd.concat(horizons_collect)
        # add current season
        data['season'] = season    
        # add to archive
        seasons_collect.append(data)
    # concatenate across seasons
    data = pd.concat(seasons_collect).set_index(['season', 'horizon'])

    return data


from pySODM.optimization.objective_functions import ll_poisson
def make_data_pySODM_compatible(strains, use_ED_visits, start_date, end_date, season): 
    """
    A function formatting the NC Influenza data in pySODM format depending on the desire to use strain or ED visit information
    """

    if strains:
        # pySODM llp data arguments
        states = ['I_inc', 'H_inc', 'H_inc']
        log_likelihood_fnc = [ll_poisson, ll_poisson, ll_poisson]
        log_likelihood_fnc_args = [[],[],[]]
        # pySODM formatting for flu A
        flu_A = get_NC_influenza_data(start_date, end_date, season)['H_inc_A']
        flu_A = flu_A.rename('H_inc') # pd.Series needs to have matching model state's name
        flu_A = flu_A.reset_index()
        flu_A['strain'] = 0
        flu_A = flu_A.set_index(['date', 'strain']).squeeze()
        # pySODM formatting for flu B
        flu_B = get_NC_influenza_data(start_date, end_date, season)['H_inc_B']
        flu_B = flu_B.rename('H_inc') # pd.Series needs to have matching model state's name
        flu_B = flu_B.reset_index()
        flu_B['strain'] = 1
        flu_B = flu_B.set_index(['date', 'strain']).squeeze()
        # attach all datasets
        data = [get_NC_influenza_data(start_date, end_date, season)['I_inc'], flu_A, flu_B]
    else:
        # pySODM llp data arguments
        states = ['I_inc', 'H_inc']
        log_likelihood_fnc = [ll_poisson, ll_poisson]
        log_likelihood_fnc_args = [[],[]]
        # pySODM data
        data = [get_NC_influenza_data(start_date, end_date, season)['I_inc'], get_NC_influenza_data(start_date, end_date, season)['H_inc']]
    # omit I_inc
    if not use_ED_visits:
        data = data[1:]
        states = states[1:]
        log_likelihood_fnc = log_likelihood_fnc[1:]
        log_likelihood_fnc_args = log_likelihood_fnc_args[1:]

    return data, states, log_likelihood_fnc, log_likelihood_fnc_args


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

from datetime import datetime
from scipy.ndimage import gaussian_filter1d

def get_transmission_coefficient_timeseries(modifier_vector, sigma=2.5):
    """
    A function mapping the modifier_vectors between Sep 15 and May 15 and smoothing it with a gaussian filter

    input
    -----

    - modifier_vector: np.ndarray
        - 1D numpy array (time) or 2D numpy array (time x spatial unit).
        - Each entry represents a value of a knotted temporal modifier, the length of each modifier is equal to the number of days between Oct 15 and Apr 15 divided by `len(modifier_vector)`.

    - sigma: float 
        - gaussian smoother's standard deviation. higher values represent more smooth trajectories but increase runtime. None represents no smoothing (fastest).

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

def plot_fit(simout, data, states, fig_path, identifier,
                coordinates_data_also_in_model, aggregate_over, additional_axes_data):
    """
    Visualises the goodness of fit for every season
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
    for i, df in enumerate(data):
        
        # aggregate data
        for dimension in simout.dims:
            if dimension in aggregate_over[i]:
                simout = simout.sum(dim=dimension)
        
        # loop over coordinates 
        if coordinates_data_also_in_model[i]:
            for coord in coordinates_data_also_in_model[i]:
                # get dimension coord is in #TODO: LIMITED TO ONE COORDINATE PER DIMENSION PER DATASET !!!
                dim_name = additional_axes_data[i][0]
                coord = coord[0]
                # plot
                ax[k].scatter(df.index.get_level_values('date').values, 7*df.loc[slice(None), coord].values, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
                if samples:
                    ax[k].fill_between(simout['date'], 7*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=0.05/2),
                            7*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
                    ax[k].fill_between(simout['date'], 7*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=0.50/2),
                            7*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)
                else:
                    ax[k].plot(simout['date'], 7*simout[states[i]].sel({dim_name: coord}), color='blue')
                ax[k].set_title(f'State: {states[i]}; Dim: {dim_name} ({coord})')
                k += 1
        else:
            # plot
            ax[k].scatter(df.index, 7*df.values, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
            if samples:
                ax[k].fill_between(simout['date'], 7*simout[states[i]].quantile(dim='draws', q=0.05/2),
                            7*simout[states[i]].quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
                ax[k].fill_between(simout['date'], 7*simout[states[i]].quantile(dim='draws', q=0.50/2),
                            7*simout[states[i]].quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)
            else:
                ax[k].plot(simout['date'], 7*simout[states[i]], color='blue')
            ax[k].set_title(f'State: {states[i]}')
            k += 1
        
        # reset output
        simout = out_copy

    plt.tight_layout()
    plt.savefig(fig_path+f'{identifier}-FIT.pdf')
    plt.close()


