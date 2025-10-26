"""
A script optimizing the variance parameter of the GRW baseline model without drift to achieve the best (lowest) WIS to all historical NC data.
Then uses the optimal variance to compute the WIS scores for the GRW baseline model with drift based on the historical data.
Saves the baseline model's WIS scores in `~/data/interim/calibration/flatBaselineModel-accuracy.csv`.
"""

# packages needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from hierarchSIR.utils import get_influenza_data
from hierarchSIR.accuracy import simulate_geometric_random_walk, compute_WIS, get_historic_drift

# settings
location = '37'
start_baseline_month = 11 # expressed as Hubverse reference date
start_baseline_day = 15
end_baseline_month = 4
end_baseline_day = 7
seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2023-2024', '2024-2025']

###############################
## Part 1: GRW without drift ##
###############################

# optimise noise on the baseline model
## define objective function
def objective_func(sigma, start_baseline_month, start_baseline_day, end_baseline_month, end_baseline_day):
    """
    Compute the WIS score of a flat baseline model with noise `sigma` across all available seasons
    """
    
    # LOOP seasons
    collect_seasons=[]
    for season in seasons:
        ## get the data
        data = 7*get_influenza_data(datetime(int(season[0:4]), start_baseline_month, start_baseline_day) - timedelta(weeks=1),
                                       datetime(int(season[0:4])+1, end_baseline_month, end_baseline_day)+timedelta(weeks=4),
                                       location)['H_inc']
        ## LOOP weeks
        collect_weeks=[]
        for date in data.index[:-4]:
            ### CONSTRUCT baseline model
            simout = simulate_geometric_random_walk([0,0,0,0], sigma, date, data.loc[date], 10000, location)
            ### COMPUTE WIS score
            collect_weeks.append(compute_WIS(simout, data))
        ## CONCATENATE WEEKS
        collect_weeks = pd.concat(collect_weeks, axis=0)
        collect_weeks = collect_weeks.reset_index()
        collect_weeks['season'] = season
        collect_seasons.append(collect_weeks)
    # CONCATENATE SEASONS
    collect_seasons = pd.concat(collect_seasons, axis=0)
    return collect_seasons

## compute WIS in function of sigma
### compute WIS
WIS=[]
sigma = np.arange(0.20,0.50,0.01)
for s in sigma:
    print(s)
    WIS.append(objective_func(s, start_baseline_month, start_baseline_day, end_baseline_month, end_baseline_day))
WIS_sum = [sum(df['WIS']) for df in WIS]
### get maximum
sigma_optim = sigma[np.argmin(WIS_sum)]
WIS_optim = WIS[np.argmin(WIS_sum)]
### report maximum
print(f'Optimal sigma: {sigma_optim:.3f}\n')

## visualise result
fig,ax=plt.subplots(figsize=(8.3,11.7/4))
ax.plot(sigma, WIS_sum, color='black', marker='s')
ax.set_xlabel('Baseline model parameter $\\sigma$')
ax.set_ylabel('Sum of WIS')
plt.tight_layout()
plt.savefig('optimization-baseline-model.pdf')
plt.close()

## add model name
WIS_optim['model'] = 'GRW_nodrift'

############################
## Part 2: GRW with drift ##
############################

# LOOP seasons
collect_seasons=[]
for focal_season in seasons:
    ## get the current season's data
    data = 7*get_influenza_data(datetime(int(focal_season[0:4]), start_baseline_month, start_baseline_day) - timedelta(weeks=1),
                                    datetime(int(focal_season[0:4])+1, end_baseline_month, end_baseline_day)+timedelta(weeks=4),
                                    location)['H_inc']
    ## LOOP weeks
    collect_weeks=[]
    for date in data.index[:-4]:
        ### COMPUTE historical drift 
        mu_horizon = []
        for i in range(4):
            ### GET historical drift
            mu, _ = get_historic_drift(focal_season, seasons, date+timedelta(weeks=i), 2, location)
            mu_horizon.append(mu)
        ### SIMULATE baseline model
        simout = simulate_geometric_random_walk(mu_horizon, sigma_optim, date, data[date], n_sim=1000)
        ### COMPUTE WIS score
        collect_weeks.append(compute_WIS(simout, data))
    ## CONCATENATE WEEKS
    collect_weeks = pd.concat(collect_weeks, axis=0)
    collect_weeks = collect_weeks.reset_index()
    collect_weeks['season'] = focal_season
    collect_seasons.append(collect_weeks)
# CONCATENATE SEASONS
collect_seasons = pd.concat(collect_seasons, axis=0)

## add model name
collect_seasons['model'] = 'GRW_drift'

# Save results
baselineModels_accuracy = pd.concat([WIS_optim, collect_seasons], axis=0)
baselineModels_accuracy = baselineModels_accuracy.set_index(['model', 'season', 'reference_date', 'horizon'])
baselineModels_accuracy.to_csv('../../../data/interim/calibration/baselineModels-accuracy.csv')

print(baselineModels_accuracy.groupby(by=['model'])['WIS'].mean())
print(baselineModels_accuracy.groupby(by=['model', 'season'])['WIS'].mean())