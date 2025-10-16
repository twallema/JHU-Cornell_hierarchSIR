"""
A script to compute the Weighted Interval Score (WIS) accuracy metric a set of forecasts

Designed for use with the following folder structure:

MY_FOLDER
|--- flatBaselineModel-accuracy.csv
|--- compute_accuracy.py
|--- 2023-2024
    |--- end-2023-12-16
    |--- end-2023-12-23
    |--- ...
|--- 2019-2020
    |--- ...
"""

# packages needed
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from influenza_USA.NC_forecasts.utils import get_NC_influenza_data, compute_WIS

# settings
prediction_horizon_weeks = 4
quantiles_WIS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
start_month = 12
start_day = 1
end_month = 4
end_day = 7

# finding the right simulations
location = 37                           # NC FIPS code
model_name = 'JHU_IDD-hierarchSIM'      # tentative model name
seasons = [entry for entry in os.listdir(os.path.dirname(__file__)) if os.path.isdir(os.path.join(os.path.dirname(__file__), entry))]
seasons.sort()

# extract and store simulations as an xarray dataset
gather_seasons = []
for season in seasons:

    # derive start of season year
    season_start = int(season[:4])

    # get all enddates of forecasts in a given season from the folder names
    subdirectories = [entry for entry in os.listdir(os.path.join(os.path.dirname(__file__), season)) if os.path.isdir(os.path.join(os.path.join(os.path.dirname(__file__), season), entry))]
    subdirectories.sort()
    reference_dates = [datetime.strptime(subdir[4:], '%Y-%m-%d')+timedelta(days=7) for subdir in subdirectories]
    data_ends = [datetime.strptime(subdir[4:], '%Y-%m-%d') for subdir in subdirectories] 

    # only evaluate between user-supplied start and enddate
    filtered_subdirectories = []
    filtered_reference_dates = []
    for subdir, data_end, reference_date in zip(subdirectories, data_ends, reference_dates):
        if datetime(season_start, start_month, start_day) <= data_end <= datetime(season_start+1, end_month, end_day):
            filtered_subdirectories.append(subdir)
            filtered_reference_dates.append(reference_date)
    subdirectories = filtered_subdirectories
    reference_dates = filtered_reference_dates

    # loop over directories to collect groundtruth, forecasts and baseline model WIS
    datas = []
    simouts = []
    baselines = []
    for subdir, reference_date in zip(subdirectories, reference_dates):
        ## get forecast
        tmp = pd.read_csv(os.path.join(season, subdir, f'{reference_date.date()}-{model_name}.csv'), parse_dates=True, date_format='%Y-%m-%d')
        ## slice location, quantiles and right target
        tmp = tmp[((tmp['location'] == location) & (tmp['output_type'] == 'quantile') & (tmp['target'] == 'wk inc flu hosp'))]
        ## make sure ref date is datetime
        tmp['reference_date'] = pd.to_datetime(tmp['reference_date'])
        tmp['target_end_date'] = pd.to_datetime(tmp['target_end_date'])
        ## append to list
        simouts.append(tmp)
        ## get groundthruth data (+ the forecast horizon of four weeks)
        data = get_NC_influenza_data(reference_date+timedelta(weeks=-1), reference_date+timedelta(weeks=3), season)['H_inc']*7
        datas.append(data)
        ## get baseline WIS scores
        baseline = pd.read_csv('flatBaselineModel-accuracy.csv', parse_dates=True, date_format='%Y-%m-%d')
        baseline['reference_date'] = pd.to_datetime(baseline['reference_date'])
        baseline = baseline[baseline['reference_date'] == reference_date]
        baselines.append(baseline[['reference_date', 'horizon', 'WIS']])

    # make a dataframe for the output of the season
    idx = pd.MultiIndex.from_product([[season,], reference_dates, range(-1,prediction_horizon_weeks)], names=['season', 'reference_date', 'horizon'])
    season_accuracy = pd.DataFrame(index=idx, columns=['WIS', 'relative_WIS'])

    # loop over weeks
    for reference_date, simout, data, baseline in zip(reference_dates, simouts, datas, baselines):
        # compute WIS and relative WIS
        season_accuracy.loc[(season, reference_date, slice(None)), 'WIS'] = compute_WIS(simout, data).values
        season_accuracy.loc[(season, reference_date, slice(None)), 'relative_WIS'] = compute_WIS(simout, data).values / baseline['WIS'].values
    # collect season results
    gather_seasons.append(season_accuracy)

# concatenate season results
output = pd.concat(gather_seasons, axis=0)

# omit horizon -1
output = output.reset_index()
output = output[output['horizon'] != -1]
output = output.set_index(['season', 'reference_date', 'horizon'])

print(output.mean())
print(output.groupby(by=['season']).mean())
print(output.groupby(by=['season', 'horizon']).mean())

# output to csv
output.to_csv('accuracy.csv')