"""
A script to convert the raw vaccination data from 2010-2024 in a format more suitable for the model. Beware:
    - No vaccination data for 72000 (Puerto Rico) is available at all.
    - Recommend not using season 2018-2019 because no vaccination data on 34000 (NJ) and 11000 (DC) was found.
"""

############################
## Load required packages ##
############################

import os
import numpy as np
import pandas as pd

##############################
## Define a helper function ##
##############################

def determine_season(date):
    """
    Determines the influenza season a date false in
    """
    if date.month < 8:
        season = f'{int(date.year-1)}-{int(date.year)}'
    else:
        season = f'{int(date.year)}-{int(date.year+1)}'
    return season

###################################################
## Construct a dataframe with the desired format ##
###################################################

# use demography to determine what the age groups and spatial units are
#demography = pd.read_csv(os.path.join(os.getcwd(), '../../interim/demography/demography_states_2023.csv'), dtype={'fips': str}) # state, age, population
#ages = demography['age'].unique()
#states = demography['fips'].unique()
# load the data
vaccination = pd.read_csv(os.path.join(os.getcwd(), '../../raw/vaccination/vacc_alldoses_age_Flu_2024_R1_allflu_allseasons.csv'), parse_dates=True, dtype={'subpop': str})
# retain only relevant columns
vaccination = vaccination[['date_admin', 'subpop', 'age_group', 'vacc_age_daily']]
# sum over age groups + convert daily incidence to weekly incidence
vaccination = 7*vaccination.groupby(by=['date_admin', 'subpop'])['vacc_age_daily'].sum()
# drop index
vaccination = vaccination.reset_index()
# rename columns
vaccination = vaccination.rename(columns={'date_admin':'date', 'subpop': 'fips_state', 'vacc_age_daily': 'incidence'})
# make sure dates are datetime
vaccination['date'] = pd.to_datetime(vaccination['date'])
# convert fips_state from str to int
vaccination['fips_state'] = vaccination['fips_state'].apply(lambda x: int(x[0:2]))
# re-introduce index
vaccination = vaccination.set_index(['date', 'fips_state'])
# take cumulative sum
vaccination['cumulative'] = vaccination.groupby(by=['fips_state'])['incidence'].cumsum()
# re-introduce season
vaccination = vaccination.reset_index()
vaccination['season'] = vaccination['date'].apply(lambda x: determine_season(x))

#################
## Save result ##
#################

vaccination.to_csv(os.path.join(os.getcwd(),'../../interim/vaccination/vaccination_incidences_2010-2024.csv'), index=False)
