"""
A script to make visualisations of the Weighted Interval Score (WIS) accuracy metric for CDC FluSight models

Uses the output of `compute-accuracy_flusight.py` as input.
"""

# packages needed
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gmean
from datetime import timedelta

# settings
baseline = 'FluSight-baseline'
objective = 'WIS'
log = False         # TRUE: log transform WIS scores before running the computation
mean = True        # FALSE: compute rel. WIS in each territory using gmean
glob = False        # FALSE: compute rel. WIS per territory and then take the mean across territories

# load in data
df = pd.read_csv('accuracy.csv', dtype={'location': str})
df['reference_date'] = pd.to_datetime(df['reference_date'])

# omit US
df = df[df['location'] != 'US']

# compute the fraction of NaNs
nan_fraction_per_model = (
    df
    .assign(is_nan=df[objective].isna())
    .groupby("model")["is_nan"]
    .mean()
)

# exclude if fraction of NaNs is higher than 5%
df = df[df['model'].isin(nan_fraction_per_model[nan_fraction_per_model <= 0.15].index.values)]

# log score?
if log:
    df[objective] = np.log(df[objective])

# compute global WIS at once
ref_dates = np.sort(df['reference_date'].unique())
results = []
if glob:
    for i, ref_date in enumerate(ref_dates):
        # consider all reference dates up to current
        current_window = ref_dates[:i+1]
        # subset dataframe
        df_window = df[df['reference_date'].isin(current_window)]
        # sum all WIS scores per model
        if not mean:
            WIS_sum = df_window.groupby(by='model')[objective].apply(lambda x: np.exp(np.sum(np.log(x))/len(x)) ).to_frame()
        else:
            WIS_sum = df_window.groupby(by='model')[objective].sum().to_frame()
        # normalise with the baseline
        WIS_sum[f'rel_{objective}'] = WIS_sum / WIS_sum.loc[baseline]
        WIS_sum = WIS_sum.reset_index()
        # attach current "as-of" reference date
        WIS_sum['as_of'] = ref_date
        results.append(WIS_sum)
    # combine all weeks
    df = pd.concat(results, ignore_index=True)
    # compute ranking
    ranking = df[df['as_of'] == max(df['as_of']) - timedelta(weeks=11)]
    ranking = ranking.sort_values(f'rel_{objective}')
else:
    # compute rel WIS per territory and then WIS over it
    for i, ref_date in enumerate(ref_dates):
        # consider all reference dates up to current
        current_window = ref_dates[:i+1]
        # subset dataframe
        df_window = df[df['reference_date'].isin(current_window)]
        # sum all WIS scores per model
        if not mean:
            WIS_sum = df_window.groupby(by=['model', 'location'])[objective].apply(lambda x: np.exp(np.sum(np.log(x))/len(x)) ).to_frame()
        else:
            WIS_sum = df_window.groupby(by=['model', 'location'])[objective].sum().to_frame()
        # normalise with the baseline
        WIS_sum[f'rel_{objective}'] = WIS_sum / WIS_sum.loc[baseline]
        # mean over locations
        #WIS_sum = WIS_sum.groupby(by='model')[f'rel_{objective}'].mean()
        WIS_sum = WIS_sum.reset_index() 
        # attach current "as-of" reference date
        WIS_sum['as_of'] = ref_date
        results.append(WIS_sum)
    # combine all weeks
    df = pd.concat(results, ignore_index=True)
    # show score by location of our model
    ranking = df[((df['model'] == 'Cornell_JHU-hierarchSIR') & (df['as_of'] == max(df['as_of']) - timedelta(weeks=1)))]
    ranking = ranking.sort_values(f'rel_{objective}')
    # load demography and merge to ranking
    demo = pd.read_csv('../../../data/interim/demography/demography.csv')
    ranking["location"] = ranking["location"].astype(int)
    ranking = ranking.merge(demo[["fips_state", "population"]], left_on="location", right_on="fips_state", how="left")
    # excluding California, Texas, NY, FL only increases R2 to 0.1
    # perform linear regression
    x = ranking["population"].values
    y = ranking["rel_WIS"].values
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    print("Intercept:", intercept)
    print("Slope:", slope)
    print("R^2:", r2)
    
    # average over locations
    df = df.groupby(by=['model', 'as_of'])[f'rel_{objective}'].mean().reset_index()

import sys
sys.exit()

# pre-format legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D(
        [0], [0],
        color='grey',
        marker='o',
        linestyle='-',
        markerfacecolor='none',   # hollow marker
        markeredgecolor='grey',
        label='Other FluSight models'
    ),
    Line2D(
        [0], [0],
        color='hotpink',
        marker='o',
        linestyle='-',
        markerfacecolor='hotpink',  # filled marker
        markeredgecolor='hotpink',
        label='Cornell_JHU-hierarchSIR'
    )
]

# limit data to match Sore's visuals
df = df[df['as_of'] <= datetime(2026,3,21)]

# visualise results (non-compressed)
fig,ax=plt.subplots(figsize=(8.27/1.32, 11.69/4))
for mn in df['model'].unique():
    # plot them all
    if ((mn != baseline) & (mn != 'Cornell_JHU-hierarchSIR')):
        ax.plot(df['as_of'].unique(), df[df['model'] == mn][f'rel_{objective}'].values, color='black', alpha=0.1, marker='o', markerfacecolor='none', label=mn)
for mn in ['Cornell_JHU-hierarchSIR']:
    ax.plot(df['as_of'].unique(), df[df['model'] == mn][f'rel_{objective}'].values, color='hotpink', alpha=1, marker='o', label=mn)
    
# format axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
## X
ax.set_xticks([datetime(2025,12,1), datetime(2026,1,1), datetime(2026,2,1), datetime(2026,3,1)])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
## Y
if log:
    ax.set_ylim([0.8,1.2])
    ax.set_yticks([0.9, 1.0, 1.1])
    ax.set_ylabel(f'log. rel. {objective} (-)')
else:
    ax.set_ylim([0.4,1.6])
    ax.set_yticks([0.5, 0.75, 1.0, 1.25])
    ax.set_ylabel(f'rel. {objective} (-)')
ax.legend(handles=legend_elements, frameon=False, loc='upper right')

fig.tight_layout()
fig.savefig(f'accuracy_flusight_log-{log}_mean-{mean}_{objective}.pdf')
plt.close()


# visualise results (compressed)
fig,ax=plt.subplots(figsize=(8.27/1.32, 11.69/4/2.5))
save = []
for mn in df['model'].unique():
    # plot them all
    if ((mn != baseline) & (mn != 'Cornell_JHU-hierarchSIR')):
        # save them
        save.append(df[df['model'] == mn][f'rel_{objective}'].values)
# compute summary statistics
save = np.stack(save, axis=1)
min = np.min(save, axis=1)
max = np.max(save, axis=1)
q25 = np.quantile(save, q=0.25, axis=1)
q75 = np.quantile(save, q=0.75, axis=1)
# visualise
ax.fill_between(df['as_of'].unique(), min, max, color='black', alpha=0.1)
ax.fill_between(df['as_of'].unique(), q25, q75, color='black', alpha=0.1)

for mn in ['Cornell_JHU-hierarchSIR']:
    ax.plot(df['as_of'].unique(), df[df['model'] == mn][f'rel_{objective}'].values, color='hotpink', alpha=1, marker='o', label=mn)
    
# format axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
## X
ax.set_xticks([datetime(2025,12,1), datetime(2026,1,1), datetime(2026,2,1), datetime(2026,3,1)])
ax.set_xticklabels([])
## Y
if log:
    ax.set_ylim([0.75,1.25])
    ax.set_yticks([0.8, 1.0, 1.2])
    ax.set_ylabel(f'rel. log. {objective}')
else:
    ax.set_ylim([0.25,2.45])
    ax.set_yticks([0.75, 1.5, 2.25])
    ax.set_ylabel(f'rel. {objective}\n(mean)')
#ax.legend(handles=legend_elements, frameon=False, loc='upper right')
fig.tight_layout()
fig.savefig(f'accuracy_flusight_compressed_log-{log}_mean-{mean}_{objective}.pdf')
plt.close()

