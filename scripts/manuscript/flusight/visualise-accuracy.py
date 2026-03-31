"""
A script to make visualisations of the Weighted Interval Score (WIS) accuracy metric for CDC FluSight models

Uses the output of `compute-accuracy_flusight.py` as input.
"""

# packages needed
import os
import numpy as np
import pandas as pd
from scipy.stats import gmean
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# define baseline model
log = False
baseline = 'FluSight-baseline'

# load in data
df = pd.read_csv('accuracy.csv', dtype={'location': str})
df['reference_date'] = pd.to_datetime(df['reference_date'])

# omit US
df = df[df['location'] != 'US']

# compute the fraction of NaNs
nan_fraction_per_model = (
    df
    .assign(is_nan=df["WIS"].isna())
    .groupby("model")["is_nan"]
    .mean()
)

# exclude if fraction of NaNs is higher than 5%
df = df[df['model'].isin(nan_fraction_per_model[nan_fraction_per_model <= 0.15].index.values)]

# log score?
if log:
    df['WIS'] = np.log(df['WIS'])

# compute rolling rel WIS
ref_dates = np.sort(df['reference_date'].unique())
results = []
for i, ref_date in enumerate(ref_dates):
    # consider all reference dates up to current
    current_window = ref_dates[:i+1]
    # subset dataframe
    df_window = df[df['reference_date'].isin(current_window)]
    # sum all WIS scores per model
    WIS_sum = df_window.groupby(by='model')['WIS'].sum().to_frame()
    # normalise with the baseline
    WIS_sum['rel_WIS'] = WIS_sum / WIS_sum.loc[baseline]
    WIS_sum = WIS_sum.reset_index()
    # attach current "as-of" reference date
    WIS_sum['as_of'] = ref_date
    results.append(WIS_sum)
# combine all weeks
df = pd.concat(results, ignore_index=True)


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

# visualise results
fig,ax=plt.subplots(figsize=(8.3, 11.7/4))
for mn in df['model'].unique():
    # plot them all
    if ((mn != baseline) & (mn != 'Cornell_JHU-hierarchSIR')):
        ax.plot(df['as_of'].unique(), df[df['model'] == mn]['rel_WIS'].values, color='black', alpha=0.1, marker='o', markerfacecolor='none', label=mn)
for mn in ['Cornell_JHU-hierarchSIR']:
    ax.plot(df['as_of'].unique(), df[df['model'] == mn]['rel_WIS'].values, color='hotpink', alpha=1, marker='o', label=mn)
    
# format axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
## X
ax.set_xticks([datetime(2025,12,15), datetime(2026,1,15), datetime(2026,2,15), datetime(2026,3,15)])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
## Y
if log:
    ax.set_ylim([0.8,1.2])
    ax.set_yticks([0.9, 1.0, 1.1])
    ax.set_ylabel('log. rel. WIS (-)')
else:
    ax.set_ylim([0.4,1.6])
    ax.set_yticks([0.5, 0.7, 0.9, 1.1])
    ax.set_ylabel('rel. WIS (-)')
ax.legend(handles=legend_elements, frameon=False, loc='upper right')

fig.tight_layout()
fig.savefig('accuracy_flusight.pdf')
plt.close()
