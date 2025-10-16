"""
A script to compute the Weighted Interval Score (WIS) of FluSight contributions
"""

# packages needed
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from influenza_USA.NC_forecasts.utils import get_NC_NHSN_data, compute_WIS

# settings
ratio = 1.6 # or None
prediction_horizon_weeks = 4
quantiles_WIS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
start_month = 12    # as Hubverse reference date = data end date + 1 week
start_day = 7
end_month = 4
end_day = 7
excluded_models = ['FluSight-baseline_cat',
                   'FluSight-ens_q_cat',
                   'FluSight-equal_cat',
                   'FluSight-national_cat',
                   'UGuelphensemble-GRYPHON',
                   'NIH-Flu_ARIMA',
                   'SigSci-CREG',
                   'CADPH-FluCAT_Ensemble',
                   'JHU_CSSE-CSSE_Ensemble',
                   ]

# finding the right simulations
location = '37'
seasons = ['2023-2024',]

# extract and store simulations as an xarray dataset
gather_seasons = []
for season in seasons:

    # derive start of season year
    season_start = int(season[:4])

    # get all the model names
    model_names = [entry for entry in os.listdir(os.path.join(season)) if os.path.isdir(os.path.join(os.path.join(season), entry))]

    # loop over model names
    gather_models = []

    for model in [model for model in model_names if model not in excluded_models]:
        
        # get all enddates of forecasts in a given season from the folder names
        filenames = [entry for entry in os.listdir(os.path.join(season, model)) if entry != '.DS_Store']
        filenames.sort()
        reference_dates = [datetime.strptime(filename[:10], '%Y-%m-%d') for filename in filenames] # as str

        # only evaluate between dates
        filtered_filenames = []
        filtered_reference_dates = []
        for filename, reference_date in zip(filenames, reference_dates):
            if datetime(season_start, start_month, start_day) <= reference_date <= datetime(season_start+1, end_month, end_day):
                filtered_filenames.append(filename)
                filtered_reference_dates.append(reference_date)
        filenames = filtered_filenames
        reference_dates = filtered_reference_dates

        # get forecast conditional on the enddate
        datas = []
        simouts = []
        baselines = []
        for filename, reference_date in zip(filenames, reference_dates):
            ## get forecast
            tmp = pd.read_csv(os.path.join(season, model, filename), parse_dates=True, date_format='%Y-%m-%d')
            ## slice location, quantiles and right target
            tmp = tmp[((tmp['location'] == location) & (tmp['output_type'] == 'quantile') & (tmp['target'] == 'wk inc flu hosp'))]
            ## cut out quantiles 'increase', etc.
            tmp = tmp[~tmp['output_type_id'].isin(['decrease', 'increase', 'large_decrease', 'large_increase', 'stable'])]
            tmp['output_type_id'] = tmp['output_type_id'].astype(float)
            ## make sure ref date is datetime
            tmp['reference_date'] = pd.to_datetime(tmp['reference_date'])
            tmp['target_end_date'] = pd.to_datetime(tmp['target_end_date'])
            ## convert to magnitude of NC detect data
            if ratio:
                tmp['value'] *= ratio
            ## drop horizon = -1
            tmp = tmp[tmp['horizon'] != -1]
            ## append to list
            simouts.append(tmp)
            ## get groundthruth data (+ the forecast horizon of four weeks)
            datas.append(get_NC_NHSN_data(reference_date, reference_date+timedelta(weeks=3)))
            ## get baseline WIS scores
            baseline = pd.read_csv('flatBaselineModel-accuracy.csv', parse_dates=True, date_format='%Y-%m-%d')
            baseline['reference_date'] = pd.to_datetime(baseline['reference_date'])
            baseline = baseline[((baseline['reference_date'] == reference_date) & (baseline['horizon'] != -1))]
            baselines.append(baseline[['reference_date', 'horizon', 'WIS']])

        # make a dataframe for the output per season per model (no horizon -1 available)
        idx = pd.MultiIndex.from_product([[season,], [model], reference_dates, range(0,prediction_horizon_weeks)], names=['season', 'model', 'reference_date', 'horizon'])
        season_accuracy = pd.DataFrame(index=idx, columns=['WIS', 'relative_WIS'])

        # loop over weeks & compute WIS
        for reference_date, simout, data, baseline in zip(reference_dates, simouts, datas, baselines):
            # compute WIS and relative WIS
            season_accuracy.loc[(season, model, reference_date, slice(None)), 'WIS'] = compute_WIS(simout, data).values
            season_accuracy.loc[(season, model, reference_date, slice(None)), 'relative_WIS'] = compute_WIS(simout, data).values / baseline['WIS'].values
        
        # collect different model results per season
        gather_models.append(season_accuracy)
    # gather different seasons
    gather_seasons.append(pd.concat(gather_models, axis=0))
# concatenate different seasons
output = pd.concat(gather_seasons, axis=0)

# force numeric types
output['WIS'] = pd.to_numeric(output['WIS'])
output['relative_WIS'] = pd.to_numeric(output['relative_WIS'])

from scipy.stats import gmean
print(output.groupby(by=['season', 'model']).mean())
print(output.groupby(by=['season','model'])['relative_WIS'].apply(lambda x: gmean(x, axis=0)))

# output to csv
output.to_csv('accuracy.csv')