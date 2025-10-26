"""
A script to format the NC raw ED visits and admissions data into a more pleasant format
"""

############################
## Load required packages ##
############################

import os
import pandas as pd
from datetime import datetime, timedelta

# all paths relative to this file
abs_dir = os.path.dirname(__file__)

######################
## helper functions ##
######################


def get_cdc_week_saturday(year, week):
    # CDC epiweeks start on Sunday and end on Saturday
    # CDC week 1 is the week with at least 4 days in January
    # Start from Jan 4th and find the Sunday of that week
    jan4 = datetime(year, 1, 4)
    start_of_week1 = jan4 - timedelta(days=jan4.weekday() + 1)  # Move to previous Sunday

    # Add (week - 1) weeks and 6 days to get Saturday
    saturday_of_week = start_of_week1 + timedelta(weeks=week-1, days=6)
    return saturday_of_week

# define a function to map each date to its flu season
def get_season_label(date):
    year = date.year
    # if month >= September, season starts in current year
    if date.month >= 9:
        return f"{year}-{year + 1}"
    else:
        return f"{year - 1}-{year}"


#############################
## build incidence dataset ##
#############################

# load raw Hospitalisation and ILI data + convert to daily incidence
data_raw = [
    pd.read_csv(os.path.join(abs_dir, f'../../raw/cases/hosp-admissions_NC_2010-2025.csv'), index_col=0, parse_dates=True)[['flu_hosp']].squeeze()/7,  # hosp
    pd.read_csv(os.path.join(abs_dir, f'../../raw/cases/ED-visits_NC_2010-2025.csv'), index_col=0, parse_dates=True)[['flu_ED']].squeeze()/7               # ILI
        ]   
# rename 
data_raw[0] = data_raw[0].rename('H_inc')
data_raw[1] = data_raw[1].rename('I_inc')
# merge
data_raw = pd.concat(data_raw, axis=1)
# change index name
data_raw.index.name = 'date'
# load subtype data flu A vs. flu B
df_subtype = pd.read_csv(os.path.join(abs_dir, f'../../interim/cases/subtypes_NC_14-25.csv'), index_col=1, parse_dates=True)[['flu_A', 'flu_B']]
# merge with the epi data
df_merged = pd.merge(data_raw, df_subtype, how='outer', left_on='date', right_on='date')
# assume a 50/50 ratio where no subtype data is available
df_merged[['flu_A', 'flu_B']] = df_merged[['flu_A', 'flu_B']].fillna(1)
# compute fraction of Flu A
df_merged['fraction_A'] = df_merged['flu_A'] / (df_merged['flu_A'] + df_merged['flu_B']) # compute percent A
# re-compute flu A and flu B cases
df_merged['H_inc_A'] = df_merged['H_inc'] * df_merged['fraction_A']
df_merged['H_inc_B'] = df_merged['H_inc'] * (1-df_merged['fraction_A'])
# throw out rows with na
df_merged = df_merged.dropna()
# throw out `fraction_A`
df = df_merged[['H_inc', 'I_inc', 'H_inc_A', 'H_inc_B']]
# load FluVIEW subtype data to get flu A (H1) vs. flu A (H3)
df_subtype = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../interim/cases/subtypes_FluVIEW-interactive_14-25.csv'))
# select South HHS region
df_subtype = df_subtype[df_subtype['REGION'] == 'Region 4']
# convert year + week to a date YYYY-MM-DD index on Saturday
df_subtype['date'] = df_subtype.apply(lambda row: get_cdc_week_saturday(row['YEAR'], row['WEEK']), axis=1)
# compute ratios of Flu A (H1) / Flu A (H3)
df_subtype['ratio_H1'] = df_subtype['A (H1)'] / (df_subtype['A (H1)'] + df_subtype['A (H3)'])
# what if there is no flu A (H1) or flu A (H3)
df_subtype['ratio_H1'] = df_subtype['ratio_H1'].fillna(1)
# retain only relevant columns
df_subtype = df_subtype[['date', 'ratio_H1']].set_index('date').squeeze()
# merge with dataset
df_merged = df.merge(df_subtype, left_index=True, right_index=True, how='left')
# compute A (H1) and A (H3)
df_merged['H_inc_AH1'] = df_merged['H_inc_A'] * df_merged['ratio_H1']
df_merged['H_inc_AH3'] = df_merged['H_inc_A'] * (1 - df_merged['ratio_H1'])
# add season label
df_merged["season"] = df_merged.index.map(get_season_label)
# only retain post-2014
df_merged = df_merged[['season', 'H_inc', 'I_inc', 'H_inc_A', 'H_inc_B', 'H_inc_AH1', 'H_inc_AH3']]#.loc[slice(datetime(2014,9,1), None)]
df_merged.to_csv(os.path.join(abs_dir, '../../interim/cases/incidences_37.csv'), index=True)


######################################
## build season cumulatives dataset ##
######################################

# define seasons we want output for
seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2022-2023', '2023-2024', '2024-2025']
# loop over them
seasons_collect = []
for season in seasons:
    # get the season start
    season_start = int(season[0:4])
    # go back two seasons
    horizons_collect = []
    for i in [0, -1, -2, -3]:
        # get the data
        data = df_merged.loc[slice(datetime(season_start+i,10,1), datetime(season_start+1+i,5,1))]*7
        # calculate cumulative totals
        column_sums = {
            "horizon": i,
            "H_inc": data["H_inc"].sum(),
            "I_inc": data["I_inc"].sum(),
            "H_inc_A": data["H_inc_A"].sum(),
            "H_inc_B": data["H_inc_B"].sum(),
            "H_inc_AH1": data["H_inc_AH1"].sum(),
            "H_inc_AH3": data["H_inc_AH3"].sum(),
        }
        # create the DataFrame
        horizons_collect.append(pd.DataFrame([column_sums]))
    # concatenate data
    data = pd.concat(horizons_collect)
    # add current season
    data['season'] = season    
    # add to archive
    seasons_collect.append(data)
# concatenate across seasons
data = pd.concat(seasons_collect).set_index(['season', 'horizon'])
data.to_csv(os.path.join(abs_dir, '../../interim/cases/historic-cumulatives_37.csv'), index=True)

