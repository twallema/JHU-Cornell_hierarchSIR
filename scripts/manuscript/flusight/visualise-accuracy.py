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
from scipy.stats import linregress

# amean MAE --> False, True, True
# amean WIS --> False, True, True
# gmean WIS --> True, True, True
inclusion_threshold = 1/28

# settings
baseline = 'FluSight-baseline'
objective = 'WIS'
log = False        # TRUE: log transform WIS scores before running the computation
mean = False        # FALSE: compute rel. WIS in each territory using gmean
glob = True        # FALSE: compute rel. WIS per territory and then take the mean across territories

# load in accuracy data
df = pd.read_csv('accuracy.csv', dtype={'location': str})
df['reference_date'] = pd.to_datetime(df['reference_date'])

# limit range
df = df[df['reference_date'] <= datetime(2026,5,23)]

# load in hospital admissions data
hosp_data = pd.read_csv('target-hospital-admissions.csv')
hosp_data = hosp_data[hosp_data['location'] != 'US']
hosp_data['location'] = hosp_data['location'].astype(int)
hosp_data['date'] = pd.to_datetime(hosp_data['date'])
hosp_data = hosp_data[hosp_data['date'] > datetime(2025,10,15)]

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
df = df[df['model'].isin(nan_fraction_per_model[nan_fraction_per_model <= min(nan_fraction_per_model) + inclusion_threshold*(1-min(nan_fraction_per_model))].index.values)]

print(f"Number of included models with threshold {inclusion_threshold:.3f}: {len(df['model'].unique())}")

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
    if objective == 'WIS':
        # perform linear regression on population size versus relative WIS
        x = np.log1p(ranking["population"].values)
        y = ranking["rel_WIS"].values
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x, y)
        # do a regression between onset of the epidemic and relative WIS
        # Step 1: Find the first date per location where weekly_rate exceeds the threshold
        threshold = 4
        hosp_data = hosp_data[hosp_data['location'] != 72]
        first_cross = (hosp_data[hosp_data['weekly_rate'] > threshold].groupby('location')['date'].min().reset_index())
        first_cross.rename(columns={'date': 'first_cross_date'}, inplace=True)
        # Step 2: Normalize by the earliest date across all locations
        earliest_cross = first_cross['first_cross_date'].min()
        first_cross['days_from_earliest'] = (first_cross['first_cross_date'] - earliest_cross).dt.days
        ranking = ranking.merge(first_cross[["location", "days_from_earliest"]], left_on="location", right_on="location", how="left")
        # Step 3: Regress
        ranking_star = ranking[ranking['location'] != 72]
        x = np.log1p(ranking_star["days_from_earliest"].values)  # also significant in regular domain
        y = ranking_star["rel_WIS"].values
        slope2, intercept2, r_value2, p_value2, std_err2 = linregress(x, y)

        # Filter locations if needed
        ranking_star = ranking[ranking['location'] != 72]
        # --- Panel 1: log(days from earliest onset) vs rel_WIS ---
        x1 = np.log1p(ranking_star["days_from_earliest"].values)
        y1 = ranking_star["rel_WIS"].values
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x1, y1)
        y_pred1 = slope1 * x1 + intercept1
        # --- Panel 2: log(population) vs rel_WIS ---
        x2 = np.log(ranking_star["population"].values)
        y2 = ranking_star["rel_WIS"].values
        slope2, intercept2, r_value2, p_value2, std_err2 = linregress(x2, y2)
        y_pred2 = slope2 * x2 + intercept2
        # --- Create figure with 2 subplots ---
        fig, axes = plt.subplots(1, 2, figsize=(8.3,11.7/4), sharey=True)
        # Panel 1
        axes[0].scatter(x1, y1, color='black', label='U.S. States and Territories')
        axes[0].plot(x1, y_pred1, color='red', linewidth=2, label='Linear fit')
        axes[0].set_xlabel("log(relative onset)")
        axes[0].set_ylabel("(amean) Rel. WIS")
        axes[0].grid(alpha=0.3)
        axes[0].text(0.05, 0.95, f"Slope={slope1:.3f}\nIntercept={intercept1:.3f}\nR²={r_value1**2:.3f}\np={p_value1:.3f}",
                    transform=axes[0].transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
        # Panel 2
        axes[1].scatter(x2, y2, color='black', label='U.S. States and Territories')
        axes[1].plot(x2, y_pred2, color='red', linewidth=2, label='Linear fit')
        axes[1].set_xlabel("log(population)")
        axes[1].grid(alpha=0.3)
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].text(0.05, 0.95, f"Slope={slope2:.3f}\nIntercept={intercept2:.3f}\nR²={r_value2**2:.3f}\np={p_value2:.3f}",
                    transform=axes[1].transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
        plt.tight_layout()
        plt.savefig('regression.pdf')
        plt.close()

    # average over locations
    df = df.groupby(by=['model', 'as_of'])[f'rel_{objective}'].mean().reset_index()

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
df = df[df['as_of'] <= datetime(2026,5,23)]

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
plt.show()
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
ax.set_xticks([datetime(2025,12,1), datetime(2026,2,1), datetime(2026,4,1), datetime(2026,6,1)])
ax.set_xticklabels([])
## Y
if log:
    ax.set_ylim([0.75,1.25])
    ax.set_yticks([0.8, 1.0, 1.2])
    ax.set_ylabel(f'rel. log. {objective}')
else:
    ax.set_ylim([0.25,2.45])
    ax.set_yticks([0.70, 1.5, 2.30])
    ax.set_ylabel(f'rel. {objective}\n(mean)')
#ax.legend(handles=legend_elements, frameon=False, loc='upper right')
fig.tight_layout()
fig.savefig(f'accuracy_flusight_compressed_log-{log}_mean-{mean}_{objective}.svg')
plt.close()

