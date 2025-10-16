"""
A script to compute the Weighted Interval Score (WIS) accuracy metric a set of forecasts

Designed for use with the following folder structure:

MY_FOLDER
|--- baselineModels-accuracy.csv
|--- compute_accuracy.py
|--- SIR-1S
    |--- informed-False
        |--- immunity_linking-False
            |--- ED_visits-False
                |--- hyperpars-exclude_2014-2015
                    |--- 2014-2015 
                        |--- end-2023-12-16
                        |--- end-2023-12-23
                        |--- ...
|--- SIR-2S
    |--- ...
"""

# packages needed
import os
import numpy as np
import pandas as pd
from scipy.stats import gmean
from datetime import datetime, timedelta
from hierarchSIR.utils import get_NC_influenza_data
from hierarchSIR.accuracy import compute_WIS

# helper function
def get_subfolders(folder_path):
    subfolders = [entry for entry in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, entry))]
    subfolders.sort()
    return subfolders

# settings
prediction_horizon_weeks = 4
quantiles_WIS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
start_month = 11
start_day = 15
end_month = 4
end_day = 7
seasons_exclude = []

# finding the right simulations
model_name_overall = 'JHU_Cornell-hierarchSIR'
location = '37'                                             # NC FIPS code
model_names = get_subfolders(os.path.dirname(__file__))
informed = [False, True]
immunity_linking = [False, True]
ED_visits = [False, True]

WIS_collection = []
print('Starting loop...')
for mn in model_names:
    n_strains = int(mn[4])
    print(f'\tWorking on model: {mn}')
    for inf in informed:
        print(f'\t\tWorking on informed: {inf}')
        for il in immunity_linking:
            print(f'\t\t\tImmunity linking: {il}')
            for ev in ED_visits:
                print(f'\t\t\t\tED Visits: {ev}')
                seasons = get_subfolders(os.path.join(os.path.dirname(__file__), f'{mn}/informed-{inf}/immunity_linking-{il}/ED_visits-{ev}'))
                for season in seasons:
                    if season not in seasons_exclude:
                        # derive start of season year
                        season_start = int(season[:4])
                        # fetch hyperparameters
                        hyperparameters = get_subfolders(os.path.join(os.path.dirname(__file__), f'{mn}/informed-{inf}/immunity_linking-{il}/ED_visits-{ev}/{season}'))
                        for hp in hyperparameters:
                            # get all enddates of forecasts in a given season from the folder names
                            subdirectories = get_subfolders(os.path.join(os.path.dirname(__file__), f'{mn}/informed-{inf}/immunity_linking-{il}/ED_visits-{ev}/{season}/{hp}'))
                            reference_dates = [datetime.strptime(subdir[15:], '%Y-%m-%d') for subdir in subdirectories]
                            data_ends = [datetime.strptime(subdir[15:], '%Y-%m-%d')-timedelta(weeks=1) for subdir in subdirectories] 
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
                                tmp = pd.read_csv(os.path.join(os.path.dirname(__file__), f'{mn}/informed-{inf}/immunity_linking-{il}/ED_visits-{ev}/{season}/{hp}/{subdir}/{reference_date.date()}-{model_name_overall}.csv'), parse_dates=True, date_format='%Y-%m-%d')
                                ## slice right location and target
                                tmp = tmp[((tmp['location'] == int(location)) & (tmp['target'] == 'wk inc flu hosp'))]
                                ## sum over strains and assign to Hubverse 'value' column
                                column_names_sum = [f'strain_{i}' for i in range(n_strains)]
                                tmp['value'] = tmp[column_names_sum].sum(axis=1)
                                ## convert to quantiles if this was not available yet
                                if 'quantiles' not in tmp['output_type'].unique():
                                    ## define desired quantiles
                                    q_desired = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
                                    ## compute them 
                                    group_cols = ['reference_date', 'target', 'horizon', 'location', 'target_end_date']
                                    tmp = (
                                        tmp.groupby(group_cols)['value']
                                        .quantile(q_desired)
                                        .reset_index()
                                    )
                                    ## The 'level_5' column contains the quantile levels from the groupby-quantile
                                    tmp = tmp.rename(columns={'level_5': 'output_type_id', 'value': 'value'})
                                    tmp['output_type'] = 'quantile'
                                    ## Optional: reorder columns like original
                                    desired_col_order = ['reference_date', 'target', 'horizon', 'location',
                                                        'output_type', 'output_type_id', 'target_end_date', 'value']
                                    tmp = tmp[desired_col_order]
                                ## make sure ref date is datetime
                                tmp['reference_date'] = pd.to_datetime(tmp['reference_date'])
                                tmp['target_end_date'] = pd.to_datetime(tmp['target_end_date'])
                                ## append to list
                                simouts.append(tmp)
                                ## get groundthruth data (+ the forecast horizon of four weeks)
                                data = get_NC_influenza_data(reference_date+timedelta(weeks=-1), reference_date+timedelta(weeks=3), season)['H_inc']*7
                                datas.append(data)
                                ## get baseline WIS scores
                                baseline = pd.read_csv('baselineModels-accuracy.csv', parse_dates=True, date_format='%Y-%m-%d')
                                baseline['reference_date'] = pd.to_datetime(baseline['reference_date'])
                                baseline = baseline[baseline['reference_date'] == reference_date]
                                baselines.append(baseline[['model', 'reference_date', 'horizon', 'WIS']])

                        # make a dataframe for the output of the season
                        idx = pd.MultiIndex.from_product([[mn,], [inf], [il], [ev], [season], [hp,], reference_dates, range(-1,prediction_horizon_weeks)], names=['model', 'informed', 'immunity_linking', 'ED_visits', 'season', 'hyperparameters', 'reference_date', 'horizon'])
                        season_accuracy = pd.DataFrame(index=idx, columns=['WIS', 'relative_WIS_nodrift', 'relative_WIS_drift'])

                        # loop over weeks
                        for reference_date, simout, data, baseline in zip(reference_dates, simouts, datas, baselines):
                            # compute WIS and relative WIS
                            season_accuracy.loc[(mn, inf, il, ev, season, hp, reference_date, slice(None)), 'WIS'] = compute_WIS(simout, data).values
                            season_accuracy.loc[(mn, inf, il, ev, season, hp, reference_date, slice(None)), 'relative_WIS_nodrift'] = compute_WIS(simout, data).values / baseline[baseline['model']=='GRW_nodrift']['WIS'].values
                            season_accuracy.loc[(mn, inf, il, ev, season, hp, reference_date, slice(None)), 'relative_WIS_drift'] = compute_WIS(simout, data).values / baseline[baseline['model']=='GRW_drift']['WIS'].values
                        # collect season results
                        WIS_collection.append(season_accuracy)

output = pd.concat(WIS_collection, axis=0)

# omit horizon -1
output = output.reset_index()
output = output[output['horizon'] != -1]
output = output.set_index(['model', 'informed', 'immunity_linking', 'ED_visits',  'season', 'hyperparameters', 'reference_date', 'horizon'])

# Save output to a .csv
output.to_csv('accuracy.csv')

# Reset index to simplify
df_reset = output.reset_index()
# Columns to aggregate
cols = ['WIS', 'relative_WIS_nodrift', 'relative_WIS_drift']
group_keys = ['model', 'informed', 'immunity_linking', 'ED_visits']
# Drop NaNs
df_clean = df_reset.dropna(subset=cols)
# Group by model/immunity_linking/ED_visits
grouped = df_clean.groupby(group_keys)
# Compute mean
mean_df = grouped[cols].mean()
# Compute gmean
gmean_df = grouped[cols].agg(lambda x: gmean(np.array(x, dtype=float)))
# Add 'mean' and 'gmean' as new first-level column index
mean_df.columns = pd.MultiIndex.from_tuples([('mean', col) for col in mean_df.columns])
gmean_df.columns = pd.MultiIndex.from_tuples([('gmean', col) for col in gmean_df.columns])
# Combine them along columns
summary_df = pd.concat([mean_df, gmean_df], axis=1)
# Sort by column levels: ('mean', 'WIS'), ('gmean', 'WIS'), etc.
summary_df = summary_df.sort_index(axis=1, level=0)
summary_df = summary_df.round(2)
# Optional: save
summary_df.to_csv("accuracy_summary.csv")