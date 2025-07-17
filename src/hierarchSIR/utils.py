import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from hierarchSIR.model import imsSIR

##########################
## Model initialisation ##
##########################

def initialise_model(strains=False, immunity_linking=False, season=None, fips_state=37):
    """
    A function to intialise the hierarchSIR model

    input
    -----

    - strains: bool
        - do we want a strain-stratified model?

    - immunity_linking: bool
        - do we want to use a structure relationship to model the population's immunity?
    
    - season: str
        - what season do we want to model (only used in combination with `immunity_linking`).
    
    fips_state: int
        - '37': North Carolina
    """

    # Parameters
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
        'rho_i': np.array([0.025,]),
        'rho_h': np.array(strains*[0.025,]),
        'T_h': 3.5
        }
    
    # get inhabitants
    population = np.ones(strains) * get_demography(fips_state)

    # initialise initial condition function
    if immunity_linking:
        if strains==3:
            historic_cumulative_incidence = get_NC_cumulatives_per_season()[['H_inc_AH1', 'H_inc_AH3', 'H_inc_B']]
        elif strains==2:
            historic_cumulative_incidence = get_NC_cumulatives_per_season()[['H_inc_A', 'H_inc_B']]
        else:
            historic_cumulative_incidence = get_NC_cumulatives_per_season()['H_inc']
        ICF = initial_condition_function(population, historic_cumulative_incidence).w_immunity_linking
    else:
        historic_cumulative_incidence = get_NC_cumulatives_per_season()['H_inc']
        ICF = initial_condition_function(population, historic_cumulative_incidence).wo_immunity_linking

    # adjust parameters dictionary
    parameters['season'] = season
    if immunity_linking:
        del parameters['f_R']
        parameters['iota_1'] = parameters['iota_2'] = parameters['iota_3'] = np.ones(strains) * 1e-5

    return imsSIR(parameters, ICF, strains)


class initial_condition_function():

    def __init__(self, population, historic_cumulative_incidence):
        """
        Set up the model's initial condition function

        input
        -----

        - population: int
            - number of individuals in the modeled population.
        
        - historic_cumulative_incidence: pd.DataFrame
            - index: season, horizon. columns: I_inc, H_inc, H_inc_A, H_inc_B.
            - obtained using hierarchSIR.utils.get_NC_cumulatives_per_season
        """
        self.population = population 
        self.historic_cumulative_incidence = historic_cumulative_incidence
        pass

    def wo_immunity_linking(self, f_I, f_R, season):
        """
        A function generating the model's initial condition -- no immunity linking; direct estimation recovered population
        
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
    
    def w_immunity_linking(self, f_I, iota_1, iota_2, iota_3,  season):
        """
        A function setting the model's initial condition.
        
        input
        -----

        - f_I: float
            - Fraction of the population initially infected
        
        - iota_n: float
            - Relative influence of cumulative cases n seasons ago -- used to compute the fraction of the population initially immune


        output
        ------

        - initial_condition: dict
            - Keys: 'S0', 'I0', 'R0'. Values: int.
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
    demography = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/interim/demography/demography.csv'))

    return int(demography[demography['fips_state'] == fips_state]['population'].values[0])


################################
## Data and output formatting ##
################################

def get_NC_influenza_data(startdate: datetime,
                          enddate: datetime,
                          season: str) -> pd.DataFrame:
    """
    Get the North Carolina Influenza dataset -- containing ED visits, ED admissions and all subtype information -- for a given season

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
        - index: 'date' [datetime], columns: 'H_inc', 'I_inc', 'H_inc_A', 'H_inc_B', 'H_inc_AH1', 'H_inc_AH3' (frequency: weekly, converted to daily)
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
    df = df_merged[['H_inc', 'I_inc', 'H_inc_A', 'H_inc_B']].loc[slice(startdate,enddate)]
    # load FluVIEW subtype data to get flu A (H1) vs. flu A (H3)
    df_subtype = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/interim/cases/subtypes_FluVIEW-interactive_14-25.csv'))
    # select South HHS region
    df_subtype = df_subtype[df_subtype['REGION'] == 'Region 4']
    # convert year + week to a date YYYY-MM-DD index on Saturday
    df_subtype['date'] = df_subtype.apply(lambda row: get_cdc_week_saturday(row['YEAR'], row['WEEK']), axis=1)
    # compute ratios of Flu A (H1) / Flu A (H3)
    df_subtype['ratio_H1'] = df_subtype['A (H1)'] / (df_subtype['A (H1)'] + df_subtype['A (H3)'])
    # what if there is no flu A (H1) or flu A (H3)
    df_subtype['ratio_H1'] = df_subtype['ratio_H1'].fillna(1)
    # retain only relevant columns
    df_subtype = df_subtype[['date', 'ratio_H1']].set_index('date').squeeze()
    # merge with dataset
    df_merged = df.merge(df_subtype, left_index=True, right_index=True, how='left')
    # compute A (H1) and A (H3)
    df_merged['H_inc_AH1'] = df_merged['H_inc_A'] * df_merged['ratio_H1']
    df_merged['H_inc_AH3'] = df_merged['H_inc_A'] * (1 - df_merged['ratio_H1'])
    return df_merged[['H_inc', 'I_inc', 'H_inc_A', 'H_inc_B', 'H_inc_AH1', 'H_inc_AH3']]

def get_cdc_week_saturday(year, week):
    # CDC epiweeks start on Sunday and end on Saturday
    # CDC week 1 is the week with at least 4 days in January
    # Start from Jan 4th and find the Sunday of that week
    jan4 = datetime(year, 1, 4)
    start_of_week1 = jan4 - timedelta(days=jan4.weekday() + 1)  # Move to previous Sunday

    # Add (week - 1) weeks and 6 days to get Saturday
    saturday_of_week = start_of_week1 + timedelta(weeks=week-1, days=6)
    return saturday_of_week

def get_NC_cumulatives_per_season() -> pd.DataFrame:
    """
    A function that returns, for each season, the cumulative total incidence in the season - 0, season - 1 and season - 2.

    output
    ------

    cumulatives: pd.DataFrame
        index: season, horizon. columns: I_inc, H_inc, H_inc_A, H_inc_B, H_inc_AH1, H_inc_AH3.
    """
    # define seasons we want output for
    seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2022-2023', '2023-2024', '2024-2025']

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
                "H_inc_AH1": data["H_inc_AH1"].sum(),
                "H_inc_AH3": data["H_inc_AH3"].sum(),
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
def make_data_pySODM_compatible(strains: int,
                                use_ED_visits: bool,
                                start_date: datetime,
                                end_date: datetime,
                                season: str): 
    """
    A function formatting the NC Influenza data in pySODM format depending on the desire to use strain or ED visit information

    
    input:
    ------

    - strains: int
        - how many strains are modeled? 1: flu, 2: flu A, flu B, 3: flu A H1, flu A H3, flu B.

    - use_ED_visits: bool
        - do we want to calibrate to the ED visit stream?

    - start_date: datetime
        - desired startdate of data

    - end_date: datetime
        - desired enddate of data
    
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
    if strains == 3:
        # pySODM llp data arguments
        states = ['I_inc', 'H_inc', 'H_inc', 'H_inc', 'H_inc']
        log_likelihood_fnc = len(states) * [ll_poisson,]
        log_likelihood_fnc_args = len(states) * [[],]
        # pySODM formatting for flu A H1
        flu_AH1 = get_NC_influenza_data(start_date, end_date, season)['H_inc_AH1']
        flu_AH1 = flu_AH1.rename('H_inc') # pd.Series needs to have matching model state's name
        flu_AH1 = flu_AH1.reset_index()
        flu_AH1['strain'] = 0
        flu_AH1 = flu_AH1.set_index(['date', 'strain']).squeeze()
        # pySODM formatting for flu A H3
        flu_AH3 = get_NC_influenza_data(start_date, end_date, season)['H_inc_AH3']
        flu_AH3 = flu_AH3.rename('H_inc') # pd.Series needs to have matching model state's name
        flu_AH3 = flu_AH3.reset_index()
        flu_AH3['strain'] = 1
        flu_AH3 = flu_AH3.set_index(['date', 'strain']).squeeze()
        # pySODM formatting for flu B
        flu_B = get_NC_influenza_data(start_date, end_date, season)['H_inc_B']
        flu_B = flu_B.rename('H_inc') # pd.Series needs to have matching model state's name
        flu_B = flu_B.reset_index()
        flu_B['strain'] = 2
        flu_B = flu_B.set_index(['date', 'strain']).squeeze()
        # attach all datasets
        data = [get_NC_influenza_data(start_date, end_date, season)['I_inc'], flu_AH1, flu_AH3, flu_B, get_NC_influenza_data(start_date, end_date, season)['H_inc']]
    elif strains == 2:
        # pySODM llp data arguments
        states = ['I_inc', 'H_inc', 'H_inc', 'H_inc']
        log_likelihood_fnc = len(states) * [ll_poisson,]
        log_likelihood_fnc_args = len(states) * [[],]
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
        data = [get_NC_influenza_data(start_date, end_date, season)['I_inc'], flu_A, flu_B, get_NC_influenza_data(start_date, end_date, season)['H_inc']]
    elif strains == 1:
        # pySODM llp data arguments
        states = ['I_inc', 'H_inc', 'H_inc']
        log_likelihood_fnc = len(states) * [ll_poisson,]
        log_likelihood_fnc_args = len(states) * [[],]
        # pySODM data
        data = [get_NC_influenza_data(start_date, end_date, season)['I_inc'], get_NC_influenza_data(start_date, end_date, season)['H_inc'], get_NC_influenza_data(start_date, end_date, season)['H_inc']]
    # omit I_inc
    if not use_ED_visits:
        data = data[1:]
        states = states[1:]
        log_likelihood_fnc = log_likelihood_fnc[1:]
        log_likelihood_fnc_args = log_likelihood_fnc_args[1:]

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
        df.to_csv(path+reference_date.strftime('%Y-%m-%d')+'-JHU_IDD'+'-hierarchSIM.csv', index=False)

    return df


from pySODM.optimization.objective_functions import log_prior_normal, log_prior_lognormal, log_prior_uniform, log_prior_gamma, log_prior_normal
def get_priors(model_name, strains, immunity_linking, use_ED_visits, hyperparameters):
    """
    A function to help prepare the pySODM-compatible priors
    """
    if not immunity_linking:
        pars = ['rho_i', 'T_h', 'rho_h', 'f_R', 'f_I', 'beta', 'delta_beta_temporal']                                      # parameters to calibrate
        bounds = [(1e-3,0.075), (0.5, 14), (0.0001,0.01), (0.10,0.50), (1e-6,0.001), (0.30,0.60), (-0.40,0.40)]          # parameter bounds
        labels = [r'$\rho_{i}$', r'$T_h$', r'$\rho_{h}$',  r'$f_{R}$', r'$f_{I}$', r'$\beta$', r'$\Delta \beta_{t}$']      # labels in output figures
        # UNINFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if not hyperparameters:
            # assign priors (R0 ~ N(1.6, 0.2); modifiers nudged to zero; all others uninformative)
            log_prior_prob_fcn = 5*[log_prior_uniform,] + 2*[log_prior_normal,]
            log_prior_prob_fcn_args = [{'bounds':  bounds[0]},
                                       {'bounds':  bounds[1]},
                                       {'bounds':  bounds[2]},
                                       {'bounds':  bounds[3]},
                                       {'bounds':  bounds[4]},
                                       {'avg':  0.455, 'stdev': 0.057},
                                       {'avg':  0, 'stdev': 0.10}]
        # INFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        else:
            # load and select priors
            priors = pd.read_csv('../../data/interim/calibration/hyperparameters.csv')
            priors = priors.loc[((priors['model'] == model_name) & (priors['immunity_linking'] == immunity_linking) & (priors['use_ED_visits'] == use_ED_visits)), (['parameter', f'{hyperparameters}'])].set_index('parameter').squeeze()
            # assign values
            if strains == 1:
                log_prior_prob_fcn = 3*[log_prior_lognormal,] + 1*[log_prior_normal,] + 1*[log_prior_lognormal,] + 1*[log_prior_normal,] + 12*[log_prior_normal,] 
                log_prior_prob_fcn_args = [ 
                                        # ED visits
                                        {'s': priors['rho_i_s'], 'scale': priors['rho_i_scale']},                                       # rho_i
                                        {'s': priors['T_h_s'], 'scale': priors['T_h_scale']},                                           # T_h
                                        # >>>>>>>>>
                                        {'s': priors['rho_h_s'], 'scale': priors['rho_h_scale']},                                       # rho_h
                                        {'avg': priors['f_R_mu'], 'stdev': priors['f_R_sigma']},                                        # f_R
                                        {'s': priors['f_I_s'], 'scale': priors['f_I_scale']},                                           # f_I
                                        {'avg': priors['beta_mu'], 'stdev': priors['beta_sigma']},                                      # beta
                                        {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                        {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
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
            elif strains == 2:
                log_prior_prob_fcn = 4*[log_prior_lognormal,] + 2*[log_prior_normal,] + 2*[log_prior_lognormal,] + 2*[log_prior_normal,] + 12*[log_prior_normal,] 
                log_prior_prob_fcn_args = [ 
                                        # ED visits
                                        {'s': priors['rho_i_s'], 'scale': priors['rho_i_scale']},                                       # rho_i
                                        {'s': priors['T_h_s'], 'scale': priors['T_h_scale']},                                           # T_h
                                        # >>>>>>>>>
                                        {'s': priors['rho_h_s_0'], 'scale': priors['rho_h_scale_0']},                                   # rho_h_0
                                        {'s': priors['rho_h_s_1'], 'scale': priors['rho_h_scale_1']},                                   # rho_h_1
                                        {'avg': priors['f_R_mu_0'], 'stdev': priors['f_R_sigma_0']},                                    # f_R_0
                                        {'avg': priors['f_R_mu_1'], 'stdev': priors['f_R_sigma_1']},                                    # f_R_1
                                        {'s': priors['f_I_s_0'], 'scale': priors['f_I_scale_0']},                                       # f_I_0
                                        {'s': priors['f_I_s_1'], 'scale': priors['f_I_scale_1']},                                       # f_I_1
                                        {'avg': priors['beta_mu_0'], 'stdev': priors['beta_sigma_0']},                                  # beta_0
                                        {'avg': priors['beta_mu_1'], 'stdev': priors['beta_sigma_1']},                                  # beta_1
                                        {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                        {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
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
            elif strains == 3:
                log_prior_prob_fcn = 5*[log_prior_lognormal,] + 3*[log_prior_normal,] + 3*[log_prior_lognormal,] + 3*[log_prior_normal,] + 12*[log_prior_normal,] 
                log_prior_prob_fcn_args = [ 
                                        # ED visits
                                        {'s': priors['rho_i_s'], 'scale': priors['rho_i_scale']},                                       # rho_i
                                        {'s': priors['T_h_s'], 'scale': priors['T_h_scale']},                                           # T_h
                                        # >>>>>>>>>
                                        {'s': priors['rho_h_s_0'], 'scale': priors['rho_h_scale_0']},                                   # rho_h_0
                                        {'s': priors['rho_h_s_1'], 'scale': priors['rho_h_scale_1']},                                   # rho_h_1
                                        {'s': priors['rho_h_s_2'], 'scale': priors['rho_h_scale_2']},                                   # rho_h_2
                                        {'avg': priors['f_R_mu_0'], 'stdev': priors['f_R_sigma_0']},                                    # f_R_0
                                        {'avg': priors['f_R_mu_1'], 'stdev': priors['f_R_sigma_1']},                                    # f_R_1
                                        {'avg': priors['f_R_mu_2'], 'stdev': priors['f_R_sigma_2']},                                    # f_R_2
                                        {'s': priors['f_I_s_0'], 'scale': priors['f_I_scale_0']},                                       # f_I_0
                                        {'s': priors['f_I_s_1'], 'scale': priors['f_I_scale_1']},                                       # f_I_1
                                        {'s': priors['f_I_s_2'], 'scale': priors['f_I_scale_2']},                                       # f_I_2
                                        {'avg': priors['beta_mu_0'], 'stdev': priors['beta_sigma_0']},                                  # beta_0
                                        {'avg': priors['beta_mu_1'], 'stdev': priors['beta_sigma_1']},                                  # beta_1
                                        {'avg': priors['beta_mu_2'], 'stdev': priors['beta_sigma_2']},                                  # beta_2
                                        {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                        {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
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

    else:
        pars = ['rho_i', 'T_h', 'rho_h', 'iota_1', 'iota_2', 'iota_3', 'f_I', 'beta', 'delta_beta_temporal']                                            # parameters to calibrate
        bounds = [(1e-3,0.075), (0.5, 14), (0.0001,0.01), (0,0.001), (0,0.001), (0,0.001), (1e-6,0.001), (0.30,0.60), (-0.40,0.40)]                      # parameter bounds
        labels = [r'$\rho_{i}$', r'$T_h$', r'$\rho_{h}$',  r'$\iota_1$', r'$\iota_2$', r'$\iota_3$', r'$f_{I}$', r'$\beta$', r'$\Delta \beta_{t}$']     # labels in output figures
        # UNINFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if not hyperparameters:
            # assign priors (R0 ~ N(1.6, 0.2); modifiers and immunity parameters nudged to zero; all others uninformative)
            log_prior_prob_fcn = 3*[log_prior_uniform,] + 3*[log_prior_gamma] + 1*[log_prior_uniform,] + 2*[log_prior_normal,]                                                                                   # prior probability functions
            log_prior_prob_fcn_args = [{'bounds':  bounds[0]},
                                       {'bounds':  bounds[1]},
                                       {'bounds':  bounds[2]},
                                       {'a': 1, 'loc': 0, 'scale': 2E-04},
                                       {'a': 1, 'loc': 0, 'scale': 2E-04},
                                       {'a': 1, 'loc': 0, 'scale': 2E-04},
                                       {'bounds':  bounds[6]},
                                       {'avg':  0.455, 'stdev': 0.055},
                                       {'avg':  0, 'stdev': 0.10}]
        # INFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        else:
            # load and select priors
            priors = pd.read_csv('../../data/interim/calibration/hyperparameters.csv')
            priors = priors.loc[((priors['model'] == model_name) & (priors['immunity_linking'] == immunity_linking) & (priors['use_ED_visits'] == use_ED_visits)), (['parameter', f'{hyperparameters}'])].set_index('parameter').squeeze()
            # assign values
            if strains == 1:
                log_prior_prob_fcn = 7*[log_prior_lognormal,] + 13*[log_prior_normal,] 
                log_prior_prob_fcn_args = [ 
                                        # ED visits
                                        {'s': priors['rho_i_s'], 'scale': priors['rho_i_scale']},                                       # rho_i
                                        {'s': priors['T_h_s'], 'scale': priors['T_h_scale']},                                           # T_h
                                        # >>>>>>>>>
                                        {'s': priors['rho_h_s'], 'scale': priors['rho_h_scale']},                                       # rho_h
                                        {'s': priors['iota_1_s'], 'scale': priors['iota_1_scale']},                                     # iota_1
                                        {'s': priors['iota_2_s'], 'scale': priors['iota_2_scale']},                                     # iota_2
                                        {'s': priors['iota_3_s'], 'scale': priors['iota_3_scale']},                                     # iota_3
                                        {'s': priors['f_I_s'], 'scale': priors['f_I_scale']},                                           # f_I
                                        {'avg': priors['beta_mu'], 'stdev': priors['beta_sigma']},                                      # beta
                                        {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                        {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
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
                                        ]          # arguments of prior functions
            elif strains == 2:
                log_prior_prob_fcn = 12*[log_prior_lognormal,] + 14*[log_prior_normal,]
                log_prior_prob_fcn_args = [ 
                                        # ED visits
                                        {'s': priors['rho_i_s'], 'scale': priors['rho_i_scale']},                                       # rho_i
                                        {'s': priors['T_h_s'], 'scale': priors['T_h_scale']},                                           # T_h
                                        # >>>>>>>>>
                                        {'s': priors['rho_h_s_0'], 'scale': priors['rho_h_scale_0']},                                   # rho_h_0
                                        {'s': priors['rho_h_s_1'], 'scale': priors['rho_h_scale_1']},                                   # rho_h_1
                                        {'s': priors['iota_1_s_0'], 'scale': priors['iota_1_scale_0']},                                 # iota_1_0
                                        {'s': priors['iota_1_s_1'], 'scale': priors['iota_1_scale_1']},                                 # iota_1_1
                                        {'s': priors['iota_2_s_0'], 'scale': priors['iota_2_scale_0']},                                 # iota_2_0
                                        {'s': priors['iota_2_s_1'], 'scale': priors['iota_2_scale_1']},                                 # iota_2_1
                                        {'s': priors['iota_3_s_0'], 'scale': priors['iota_3_scale_0']},                                 # iota_3_0
                                        {'s': priors['iota_3_s_1'], 'scale': priors['iota_3_scale_1']},                                 # iota_3_1
                                        {'s': priors['f_I_s_0'], 'scale': priors['f_I_scale_0']},                                       # f_I_0
                                        {'s': priors['f_I_s_1'], 'scale': priors['f_I_scale_1']},                                       # f_I_1
                                        {'avg': priors['beta_mu_0'], 'stdev': priors['beta_sigma_0']},                                  # beta_0
                                        {'avg': priors['beta_mu_1'], 'stdev': priors['beta_sigma_1']},                                  # beta_1
                                        {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                        {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
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
                                        ]          # arguments of prior functions
            elif strains == 3:
                log_prior_prob_fcn = 17*[log_prior_lognormal,] + 15*[log_prior_normal,]
                log_prior_prob_fcn_args = [ 
                                        # ED visits
                                        {'s': priors['rho_i_s'], 'scale': priors['rho_i_scale']},                                       # rho_i
                                        {'s': priors['T_h_s'], 'scale': priors['T_h_scale']},                                           # T_h
                                        # >>>>>>>>>
                                        {'s': priors['rho_h_s_0'], 'scale': priors['rho_h_scale_0']},                                   # rho_h_0
                                        {'s': priors['rho_h_s_1'], 'scale': priors['rho_h_scale_1']},                                   # rho_h_1
                                        {'s': priors['rho_h_s_2'], 'scale': priors['rho_h_scale_2']},                                   # rho_h_2
                                        {'s': priors['iota_1_s_0'], 'scale': priors['iota_1_scale_0']},                                 # iota_1_0
                                        {'s': priors['iota_1_s_1'], 'scale': priors['iota_1_scale_1']},                                 # iota_1_1
                                        {'s': priors['iota_1_s_2'], 'scale': priors['iota_1_scale_2']},                                 # iota_1_2
                                        {'s': priors['iota_2_s_0'], 'scale': priors['iota_2_scale_0']},                                 # iota_2_0
                                        {'s': priors['iota_2_s_1'], 'scale': priors['iota_2_scale_1']},                                 # iota_2_1
                                        {'s': priors['iota_2_s_2'], 'scale': priors['iota_2_scale_2']},                                 # iota_2_2
                                        {'s': priors['iota_3_s_0'], 'scale': priors['iota_3_scale_0']},                                 # iota_3_0
                                        {'s': priors['iota_3_s_1'], 'scale': priors['iota_3_scale_1']},                                 # iota_3_1
                                        {'s': priors['iota_3_s_2'], 'scale': priors['iota_3_scale_2']},                                 # iota_3_2
                                        {'s': priors['f_I_s_0'], 'scale': priors['f_I_scale_0']},                                       # f_I_0
                                        {'s': priors['f_I_s_1'], 'scale': priors['f_I_scale_1']},                                       # f_I_1
                                        {'s': priors['f_I_s_2'], 'scale': priors['f_I_scale_2']},                                       # f_I_2
                                        {'avg': priors['beta_mu_0'], 'stdev': priors['beta_sigma_0']},                                  # beta_0
                                        {'avg': priors['beta_mu_1'], 'stdev': priors['beta_sigma_1']},                                  # beta_1
                                        {'avg': priors['beta_mu_2'], 'stdev': priors['beta_sigma_2']},                                  # beta_2
                                        {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                        {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
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
                                        ]          # arguments of prior functions
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
             additional_axes_data: list) -> None:
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
                ax[k].scatter(df_calib.index.get_level_values('date').values, 7*df_calib.loc[slice(None), coord].values, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
                if not df_valid.empty:
                    ax[k].scatter(df_valid.index.get_level_values('date').values, 7*df_valid.loc[slice(None), coord].values, color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
                
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
            ax[k].scatter(df_calib.index, 7*df_calib.values, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
            if not df_valid.empty:
                ax[k].scatter(df_valid.index, 7*df_valid.values, color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
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

# helper function
def str_to_bool(value):
    """Convert string arguments to boolean (for SLURM environment variables)."""
    return value.lower() in ["true", "1", "yes"]


