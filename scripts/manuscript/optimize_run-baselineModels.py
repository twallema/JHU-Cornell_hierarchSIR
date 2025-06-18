"""
A script optimizing the flatBaselineModel to achieve the best (lowest) WIS to all historical NC data.
Saves the optimal flatBaselineModel WIS scores in `~/data/interim/calibration/flatBaselineModel-accuracy.csv`, which is used to compute relative WIS scores for the NC models.
"""

# packages needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from hierarchSIR.utils import get_NC_influenza_data
from hierarchSIR.accuracy import simulate_geometric_random_walk, compute_WIS

# settings
location = '37'
start_optimisation_month = 12 # expressed as Hubverse reference date
start_optimisation_day = 1
end_optimisation_month = 4
end_optimisation_day = 7
seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2023-2024']

# optimise noise on the baseline model
## define objective function
def objective_func(sigma, start_optimisation_month, start_optimisation_day, end_optimisation_month, end_optimisation_day):
    """
    Compute the WIS score of a flat baseline model with noise `sigma` across all available seasons
    """
    
    # LOOP seasons
    collect_seasons=[]
    for season in seasons:
        ## get the data
        data = 7*get_NC_influenza_data(datetime(int(season[0:4]), start_optimisation_month, start_optimisation_day) - timedelta(weeks=1),
                                       datetime(int(season[0:4])+1, end_optimisation_month, end_optimisation_day)+timedelta(weeks=4),
                                       season)['H_inc']
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
sigma = np.arange(0.15,0.60,0.025)
for s in sigma:
    print(s)
    WIS.append(objective_func(s, start_optimisation_month, start_optimisation_day, end_optimisation_month, end_optimisation_day))
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
plt.show()
plt.close()

## save result
WIS_optim['model'] = 'flatBaselineModel'
WIS_optim = WIS_optim.set_index(['model', 'season', 'reference_date', 'horizon'])
WIS_optim.to_csv('../../data/interim/calibration/baselineModels-accuracy.csv')