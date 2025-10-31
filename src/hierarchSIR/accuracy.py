import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress
from datetime import datetime, timedelta
from hierarchSIR.utils import get_NC_influenza_data

def compute_WIS(simout, data):
    """
    Compute the WIS of a simulation in Hubverse format `simout` on groundtruth `data`.

    Input
    -----

    - simout: pd.DataFrame
        - Simulation in Hubverse format.
        - Columns: 'reference_date', 'target', 'horizon', 'location', 'output_type', 'output_type_id', 'target_end_date', 'value'. 

    - data: pd.Series
        - Groundtruth data.

    Output
    ------

    - WIS: pd.DataFrame
        - Columns: 'reference_date', 'horizon'
    """

    # get metadata
    reference_dates = simout['reference_date'].unique()
    horizon = simout['horizon'].unique()
    quantiles = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    # pre-allocate output dataframe
    idx = pd.MultiIndex.from_product([reference_dates, horizon], names=['reference_date', 'horizon'])
    WIS = pd.Series(index=idx, name='WIS')
    for reference_date in reference_dates:
        # Loop over horizon
        for n in horizon:
            n = float(n)
            ## get date
            date = reference_date+timedelta(weeks=n)
            ## get data
            y = data.loc[date]
            ## compute IS
            IS_alpha = []
            for q in quantiles:
                # get quantiles
                try:
                    l = simout[((simout['target_end_date'] == reference_date+timedelta(weeks=n)) & (simout['output_type_id'] == q/2))]['value'].values[0]
                    u = simout[((simout['target_end_date'] == reference_date+timedelta(weeks=n)) & (simout['output_type_id'] == 1-q/2))]['value'].values[0]
                except:
                    l = np.nan
                    u = np.nan
                # compute IS
                IS = (u - l)
                if y < l:
                    IS += 2/q * (l-y)
                elif y > u:
                    IS += 2/q * (y-u)
                IS_alpha.append(IS)
            IS_alpha = np.array(IS_alpha)
            ## compute WIS & assign
            try:
                m = simout[((simout['target_end_date'] == reference_date+timedelta(weeks=n)) & (simout['output_type_id'] == 0.50))]['value'].values[0]
            except:
                m = np.nan
            WIS.loc[reference_date, n] = (1 / (len(quantiles) + 0.5)) * (0.5 * np.abs(y-m) + np.sum(0.5*np.array(quantiles) * IS_alpha))
        return WIS

def simulate_geometric_random_walk(mu, sigma, data_end_date, data_end_value, n_sim=1000, location='37'):
    """
    Simulates a geometric random walk with drift and returns its output in Hubverse format.

    Baseline model
    --------------

    - Y_t = np.log(X_t),
    - Y_{t+1} = Y_{t} + epsilon_t,
    - epsilon_t ~ N(mu, sigma**2),

    for mu = 0 the median is constant over the predicted horizon.

    Input
    -----

    - mu: list
        - Drift (in log space). List length determines forecast horizon.

    - sigma: float
        - Uncertainty on the drift (in log space).

    - data_end_date: datetime
        - The start date of the baseline model simulation.

    - data_end_value: float
        - The initial value of the baseline model simulation (at `data_end_value`).

    - n_sim: int
        - The number of stochastic realisations of the baseline model.

    - location: str
        - The first two digits of the US state FIPS code. '37' for NC. 

    Output
    ------

    - simout: pd.DataFrame
        - Simulation output in Hubverse format.
        - Columns: 'reference_date', 'target', 'horizon', 'location', 'output_type', 'output_type_id', 'target_end_date', 'value'. 
    """
    
    ## Input checks
    if not isinstance(mu, (list, np.ndarray)):
        raise TypeError('`mu` must be a list/1D np.ndarray')
    else:
        n_weeks = len(mu)

    ## Run model 
    # initialise daterange
    dates = pd.date_range(start=data_end_date, end=data_end_date+timedelta(weeks=n_weeks), freq='D')
    # expand mu from weekly to daily
    mu = [value for value in mu for _ in range(7)]
    # pre-allocate output
    output = np.zeros([len(dates), n_sim])
    # pre-allocate startpoint
    output[0,:] = np.log(data_end_value) + np.random.normal(mu[0], sigma**2, size=n_sim)
    # simulate
    for i,_ in enumerate(dates[1:]):
        output[i+1,:] = output[i,:] + np.random.normal(mu[i], sigma**2, size=n_sim)
    # transform back to linear space
    output = np.exp(output) # dates x chains

    ## Convert to Hubverse format
    ### Pre-allocate dataframe
    reference_date = data_end_date + timedelta(weeks=1)
    target = 'wk inc flu hosp'
    horizon = range(-1,4)
    output_type = 'quantiles'
    output_type_id = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
    idx = pd.MultiIndex.from_product([[reference_date,], [target,], horizon, [location,], [output_type,], output_type_id],
                                        names=['reference_date', 'target', 'horizon', 'location', 'output_type', 'output_type_id'])
    df = pd.DataFrame(index=idx, columns=['value'])
    # attach target end date
    df = df.reset_index()
    df['target_end_date'] = df.apply(lambda row: row['reference_date'] + timedelta(weeks=row['horizon']), axis=1)
    ### Interpolate baseline to daily frequency
    # put in xarray dataset
    ds = xr.Dataset({"simout": (["dates", "draws"], output)}, coords={"dates": dates, "draws": range(n_sim)})
    # interpolate to weekly frequency
    output_dates = pd.date_range(start=data_end_date, end=data_end_date+timedelta(weeks=n_weeks), freq='W-SAT')
    ds = ds.interp(dates=output_dates)
    ### Fill in dataframe
    for q in output_type_id:
        df.loc[df['output_type_id'] == q, 'value'] = ds['simout'].quantile(dim='draws', q=q).values
    return df


def get_historic_drift(focal_season, seasons, date, drift_horizon):
    """A function to compute the drift in a historical dataset over a horizon
    """
    historic_slopes=[]
    historic_slopes_std=[]
    for historic_season in [x for x in seasons if x != focal_season]:
        #### have to get right year in season (before or after Jan 1)
        year = int(historic_season[0:4])+1 if int(date.year) > int(focal_season[0:4]) else int(historic_season[0:4])
        #### handle leap years
        month, day = (3,1) if ((date.month == 2) & (date.day == 29) & (year % 4 != 0)) else (date.month, date.day)
        #### extract data
        historic_data = 7*get_NC_influenza_data(datetime(year, month, day) - timedelta(days=0),
                                datetime(year, month, day)+timedelta(weeks=drift_horizon))['H_inc'].to_frame().iloc[-drift_horizon:]
        historic_data = historic_data.reset_index()
        historic_data['horizon'] = 7*np.array((range(-drift_horizon, 0)))
        historic_data = historic_data[['horizon', 'H_inc']]
        ### get slope (scipy.stats)
        result = linregress(historic_data['horizon'].values , np.log(historic_data['H_inc'].values))
        historic_slopes.append(result.slope)
        historic_slopes_std.append(result.stderr)
    return np.mean(historic_slopes), np.mean(historic_slopes_std)