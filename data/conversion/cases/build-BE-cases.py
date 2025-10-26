"""
A script to format the BE ILI data into a more pleasant format
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


def impute_subtypes_beginning(group):
    """A function designed to impute missing data at the start of each season
    """

    # Compute cumulative sum of cases
    group = group.sort_index()
    group["cum_cases"] = group["Ncases_flu"].cumsum()

    # Find first date where cum_cases > 20
    cutoff = group.index[group["cum_cases"] > 20]
    if len(cutoff) == 0:
        # Not enough cases in this season, skip
        return group.drop(columns="cum_cases")
    cutoff_date = cutoff[0]

    # Select rows before or at cutoff where cases > 0
    mask_training = (group.index <= cutoff_date) & (group["Ncases_flu"] > 0)
    weighted_sum = (group.loc[mask_training, subtype_cols]
                    .multiply(group.loc[mask_training, "Ncases_flu"], axis=0)
                    .sum())
    total_weight = group.loc[mask_training, "Ncases_flu"].sum()
    weighted_means = weighted_sum / total_weight

    # Identify rows to impute (before cutoff, with 0 cases)
    mask_impute = (group.index < cutoff_date)
    group.loc[mask_impute, subtype_cols] = weighted_means.values

    return group.drop(columns="cum_cases")


def impute_subtypes_end(group):
    group = group.sort_index()

    # Reverse cumulative sum from the end of the season
    group["rev_cum_cases"] = group["Ncases_flu"].iloc[::-1].cumsum().iloc[::-1]

    # Find first date (from the end) where reverse cumulative cases > 10
    cutoff = group.index[group["rev_cum_cases"] > 20]
    if len(cutoff) == 0:
        return group.drop(columns="rev_cum_cases")
    cutoff_date = cutoff[-1]  # the earliest date after which last ≥10 samples occur

    # Compute weighted means using data after or at cutoff (where Ncases_flu > 0)
    mask_training = (group.index >= cutoff_date) & (group["Ncases_flu"] > 0)
    weighted_sum = (group.loc[mask_training, subtype_cols]
                    .multiply(group.loc[mask_training, "Ncases_flu"], axis=0)
                    .sum())
    total_weight = group.loc[mask_training, "Ncases_flu"].sum()
    weighted_means = weighted_sum / total_weight

    # Replace ALL values after cutoff with these weighted means
    mask_impute = (group.index > cutoff_date)
    group.loc[mask_impute, subtype_cols] = weighted_means.values

    return group.drop(columns="rev_cum_cases")


#############################
## build incidence dataset ##
#############################


# load subtype data
subtype_data = pd.read_excel(os.path.join(abs_dir, '../../raw/cases/DTA_JH_2024_update14112024.xlsx'), sheet_name='dist_flu_subtypes_inILI')
# convert to date
def yearweek_to_datetime(year_week: str) -> datetime:
    year, week = year_week.lower().split('w')
    return datetime.fromisocalendar(int(year), int(week), 1)
subtype_data['date'] = subtype_data['week'].apply(yearweek_to_datetime)
# remove yearweek
subtype_data = subtype_data.drop('week',axis=1)
# index by date
subtype_data = subtype_data.set_index('date')


# load ILI incidence data
ILI_data = pd.read_excel(os.path.join(abs_dir, '../../raw/cases/DTA_JH_2024_update14112024.xlsx'), sheet_name='Number of cases of ILI', dtype={'Yw': str})
# add a date column
def yearweek_to_datetime(year_week: str) -> datetime:
    # Extract year and week from the string
    year = int(year_week[:4])
    week = int(year_week[4:])
    return datetime.fromisocalendar(year, week, 1)
ILI_data['date'] = ILI_data['Yw'].apply(yearweek_to_datetime)
# remove unnecessary columns
ILI_data = ILI_data.drop(['Yw', 'Flu_posratio', 'week', 'year'],axis=1)
# sum cases and catchment population over the regions
ILI_data = ILI_data.groupby('date', as_index=False).sum()
# divide the number of cases by the estimated catchment population to obtain the incidence per 100 K
for new_column, number_cases, catchment_population in zip(['ILI_0to1', 'ILI_1to4', 'ILI_5to14', 'ILI_15to19', 'ILI_20to64', 'ILI_65to84', 'ILI_85to100'],
                                                        ['Number of cases ILI [< 1 year]', 'Number of cases ILI [1 - 4 year]', 'Number of cases ILI [5 - 14 year]', 'Number of cases ILI [15 - 19 year]', 'Number of cases ILI [20 - 64 year]', 'Number of cases ILI [65 - 84 year]', 'Number of cases ILI [>= 85 year]'],
                                                        ['Catchment_population_per_region_0', 'Catchment_population_per_region_14', 'Catchment_population_per_region_514', 'Catchment_population_per_region_1519', 'Catchment_population_per_region_2064', 'Catchment_population_per_region_6584', 'Catchment_population_per_region_85100']):
    ILI_data[new_column] = ILI_data[number_cases] / ILI_data[catchment_population]*100000
# compute the overall incidence
ILI_data['ILI_0to100'] = (ILI_data['Number of cases ILI [< 1 year]'] + ILI_data['Number of cases ILI [1 - 4 year]'] + ILI_data['Number of cases ILI [5 - 14 year]'] + ILI_data['Number of cases ILI [15 - 19 year]'] + ILI_data['Number of cases ILI [20 - 64 year]'] + ILI_data['Number of cases ILI [65 - 84 year]'] + ILI_data['Number of cases ILI [>= 85 year]']) / (ILI_data['Catchment_population_per_region_0'] + ILI_data['Catchment_population_per_region_14'] + ILI_data['Catchment_population_per_region_514'] + ILI_data['Catchment_population_per_region_1519'] + ILI_data['Catchment_population_per_region_2064'] + ILI_data['Catchment_population_per_region_6584'] + ILI_data['Catchment_population_per_region_85100']) * 100000
# remove all obsolete columns
ILI_data = ILI_data.drop(['region', 'N_GPpractices_per_region',]+['Number of cases ILI [< 1 year]', 'Number of cases ILI [1 - 4 year]', 'Number of cases ILI [5 - 14 year]', 'Number of cases ILI [15 - 19 year]', 'Number of cases ILI [20 - 64 year]', 'Number of cases ILI [65 - 84 year]', 'Number of cases ILI [>= 85 year]']+['Catchment_population_per_region_0', 'Catchment_population_per_region_14', 'Catchment_population_per_region_514', 'Catchment_population_per_region_1519', 'Catchment_population_per_region_2064', 'Catchment_population_per_region_6584', 'Catchment_population_per_region_85100'], axis=1)
# index by date
ILI_data = ILI_data.set_index('date')
# retain only relevant column and convert to total cases in Belgium 
ILI_data = ILI_data['ILI_0to100'] / 100000 * 11.5E6 / 7 


# merge the datasets
df_merged = ILI_data.to_frame().merge(subtype_data, left_index=True, right_index=True, how='left')
# add season label
df_merged["season"] = df_merged.index.map(get_season_label)
# replace Nan in column
df_merged['Ncases_flu'] = df_merged['Ncases_flu'].fillna(0)
# omit until 2010-2011 season as this is the first subtyped season
df_merged = df_merged.loc[slice(datetime(2010,9,1), None)]
# impute early and late season with the averages of the first and last 20 samples taken
subtype_cols = ["perc_h1n1", "perc_h3n2", "perc_flu_a_notsubtyped", "perc_yam", "perc_vic", "perc_flu_b_nolineage"]
df_merged = df_merged.groupby("season", group_keys=False).apply(impute_subtypes_beginning)
df_merged = df_merged.groupby("season", group_keys=False).apply(impute_subtypes_end)
# unpack to flu A and flu B
df_merged['I_inc'] = df_merged['ILI_0to100']
df_merged['H_inc'] = df_merged['ILI_0to100']
df_merged['H_inc_A'] = df_merged['H_inc'] * (df_merged['perc_h1n1'] + df_merged['perc_h3n2'] + df_merged['perc_flu_a_notsubtyped'])
df_merged['H_inc_B'] = df_merged['H_inc'] * (df_merged['perc_yam'] + df_merged['perc_vic'] + df_merged['perc_flu_b_nolineage'])
# unpack to flu AH1N1 and AH3N2
df_merged['H_inc_AH1'] =  df_merged['H_inc_A'] * (df_merged['perc_h1n1'] / (df_merged['perc_h1n1'] + df_merged['perc_h3n2']))
df_merged['H_inc_AH3'] =  df_merged['H_inc_A'] * (df_merged['perc_h3n2'] / (df_merged['perc_h1n1'] + df_merged['perc_h3n2']))
# if no flu A was subtyped impute 50/50
df_merged['H_inc_AH1'] = df_merged['H_inc_AH1'].fillna(0.5)
df_merged['H_inc_AH3'] = df_merged['H_inc_AH3'].fillna(0.5)
# retain only relevant columns
df_merged = df_merged[['season', 'I_inc', 'H_inc', 'H_inc_A', 'H_inc_B', 'H_inc_AH1', 'H_inc_AH3']]
df_merged.to_csv(os.path.join(abs_dir, '../../interim/cases/incidences_57.csv'), index=True)

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
data.to_csv(os.path.join(abs_dir, '../../interim/cases/historic-cumulatives_57.csv'), index=True)

