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

# load the data
vaccination = pd.read_csv(os.path.join(os.getcwd(), '../../raw/vaccination/vacc_alldoses_age_Flu_2024_R1_allflu_allseasons.csv'), parse_dates=True, dtype={'subpop': str})
# retain only relevant columns
vaccination = vaccination[['date_admin', 'subpop', 'age_group', 'vacc_age_daily', 'vacc_age']]
# sum over age groups
vaccination = vaccination.groupby(['date_admin', 'subpop'], as_index=True)[['vacc_age_daily', 'vacc_age']].sum()
# convert daily incidence to weekly incidence
vaccination['vacc_age_daily'] *= 7
# drop index
vaccination = vaccination.reset_index()
# rename columns
vaccination = vaccination.rename(columns={'date_admin':'date', 'subpop': 'fips_state', 'vacc_age_daily': 'incidence', 'vacc_age': 'cumulative'})
# make sure dates are datetime
vaccination['date'] = pd.to_datetime(vaccination['date'])
# convert fips_state from str to int
vaccination['fips_state'] = vaccination['fips_state'].apply(lambda x: int(x[0:2]))
# re-introduce season
vaccination['season'] = vaccination['date'].apply(lambda x: determine_season(x))

#################
## Save result ##
#################

vaccination.to_csv(os.path.join(os.getcwd(),'../../interim/vaccination/vaccination_incidences_2010-2024.csv'), index=False)
