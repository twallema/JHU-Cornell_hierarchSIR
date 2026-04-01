"""
A script to compute the Weighted Interval Score (WIS) and Mean Absolute Error (MAE) accuracy metric for CDC FluSight models

Designed for use with the following folder structure:

cdcepi/FluSight-forecast-hub/model-output
|--- FluSight-ensemble
|--- CADPH-FluCAT_Ensemble
    |--- 2026-03-14-CADPH-FluCAT_Ensemble.csv
    |--- 2026-03-21-CADPH-FluCAT_Ensemble.csv
    |--- ...
|--- Cornell_JHU-hierarchSIR
    |--- ...
"""

# packages needed
import os
import numpy as np
import pandas as pd
from scipy.stats import gmean
from datetime import datetime, timedelta
from hierarchSIR.accuracy import compute_WIS

# start of evaluation
eval_start_date = datetime(2025, 11, 1)
eval_end_date = datetime(2026, 5, 1)

# all paths absolute to this file
abs_dir = os.path.dirname(__file__)

# helper functions to find folders and filenames
def get_subfolders(folder_path):
    subfolders = [entry for entry in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, entry))]
    subfolders.sort()
    return subfolders

def list_files_in_directory(directory_path):
    files_list = []
    # Get all entries (files and directories) in the directory
    entries = os.listdir(directory_path)
    for entry in entries:
        # Construct the full path to check if it's a file
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files_list.append(entry)
    return files_list

# retrieve the target data
data = pd.read_csv('../target-data/target-hospital-admissions.csv', parse_dates=['date'], date_format='%Y-%m-%d')
data = data[data['date'] > eval_start_date]
data = data.sort_values(by=["date", "location"]).reset_index()[['date', 'location', 'value']]

# finding the right simulations
model_names = get_subfolders(os.path.dirname(__file__))

# exclude models that didn't participate in 2025-2026
exclude_models = ['CFA_Pyrenew-Pyrenew_E_Flu', 'CMU-climate_baseline',
                  'FluSight-base_seasonal', 'FluSight-baseline_cat', 'FluSight-dist_cat', 'FluSight-ens_q_cat', 'FluSight-ens_q_cat_sub', 'FluSight-equal_cat', 'FluSight-lop_norm', 'FluSight-national_cat',
                  'GH-model', 'GT-FluFNP', 'Gatech-ensemble_point', 'Google_SAI-FluBoostQR',
                  'ISU_NiemiLab-ENS', 'ISU_NiemiLab-GPE', 'ISU_NiemiLab-NLH', 'ISU_NiemiLab-SIR',
                  'JHUAPL-DMD', 'JHUAPL-Morris', 'Metaculus-cp', 'MetroCast-ensemble', 'MOBS-GLEAM_FLUH', 'NU_UCSD-GLEAM_AI_FLUH',
                  'NU-PGF_FLUH', 'PSI-PROF_beta', 'SGroup-RandomForest', 'SigSci-BECAM', 'SigSci-CREG', 'Stevens-GBR', 'Stevens-ILIForecast',
                  'UGA_flucast-OKeeffe', 'UVAFluX-CESGCN', 'UVAFluX-OptimWISE', 'VTSanghani-Ensemble', 'UMass-trends_ensemble', 'cfa-flumech', 'cfarenewal-cfaepimlight',
                  'fjordhest-ensemble']
model_names = [mn for mn in model_names if mn not in exclude_models]

all_locations = data['location'].unique()

# WIS computation loop
WIS_collection = []
print('Starting loop...')
mn_acc_collect = []
for mn in model_names:
    print(f'\tWorking on model: {mn}')
    filenames = list_files_in_directory(os.path.join(abs_dir, mn))
    filenames = [fn for fn in filenames if fn != '.DS_Store']
    filenames.sort()
    fn_acc_collect = []
    for fn in filenames:
        # get the reference date
        ref_date = datetime.strptime(fn[:10], "%Y-%m-%d")
        # only use if larger than the eval date
        if ((ref_date >= eval_start_date) & (ref_date <= eval_end_date)):
            # get the forecasts
            forecast = pd.read_csv(os.path.join(abs_dir, mn, fn), dtype={'location': str}, parse_dates=['reference_date', 'target_end_date'], date_format='%Y-%m-%d')
            # slice right target and metrics
            forecast = forecast[((forecast['target'] == 'wk inc flu hosp') & (forecast['output_type'] == 'quantile'))]
            forecast['output_type_id'] = forecast['output_type_id'].astype(float)
            locations = forecast['location'].unique()
            # loop over locations
            loc_acc_collect = []
            for loc in locations:
                # get the corresponding target data
                d = data[((data['date'].isin(forecast['target_end_date'].unique())) & (data['location'] == loc))][['date', 'value']].set_index('date').squeeze()
                # prevent collapse to float when there is only one value
                if isinstance(d, float):
                    d = pd.Series(index=[ref_date], data=d)
                # slice the right location in forecast
                fc = forecast[forecast['location'] == loc]
                # compute the WIS scores
                acc = compute_WIS(fc, d)
                acc = acc.reset_index()
                acc['location'] = loc
                acc['model'] = mn
                # append the AE
                fc = fc[fc['output_type_id'] == 0.50]
                fc = fc.merge(d.rename("obs"), left_on="target_end_date", right_index=True, how='left')
                acc['AE'] = np.abs((fc['value'] - fc['obs']).values)
                loc_acc_collect.append(acc)
            fn_acc_collect.append(pd.concat(loc_acc_collect, axis=0))
    mn_acc_collect.append(pd.concat(fn_acc_collect, axis=0))
mn_acc = pd.concat(mn_acc_collect, axis=0)

# omit horizon -1
mn_acc = mn_acc[mn_acc['horizon'] != -1]

# build a maximalist dataframe
all_models = mn_acc['model'].unique()
all_horizons = mn_acc['horizon'].unique()
all_locations = data['location'].unique()
all_reference_dates = mn_acc['reference_date'].unique()

index = pd.MultiIndex.from_product([all_models, all_reference_dates, all_locations, all_horizons], names=["model", "reference_date", "location", "horizon"])
df = pd.DataFrame(index=index, columns=['WIS', 'AE'])

# join the WIS data
mn_acc = mn_acc.set_index(["model", "reference_date", "location", "horizon"])
df.update(mn_acc)

# Save output to a .csv
df.to_csv('accuracy.csv')
