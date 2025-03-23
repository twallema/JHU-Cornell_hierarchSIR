"""
A script to build a list containing US state and county names and corresponding FIPS codes
"""

############################
## Load required packages ##
############################

import os
import pandas as pd

#####################
## Script settings ##
#####################



#######################
## Format FIPS codes ##
#######################

# load FIPS codes & slice relevant columns
state_FIPS = pd.read_csv(os.path.join(os.getcwd(), '../../raw/demography/national_state2020.txt'), delimiter='|')[['STATE', 'STATEFP', 'STATE_NAME']]
# the following states/territories are not present in the demographic data (notably: Puerto Rico)
remove_state_FIPS = [60, 66, 69, 72, 74, 78]
# remove undesired states and territories
state_FIPS = state_FIPS[~state_FIPS['STATEFP'].isin(remove_state_FIPS)]
# rename columns
state_FIPS = state_FIPS.rename(columns={"STATE": "abbreviation_state","STATEFP": "fips_state", "STATE_NAME": "name_state",})
# use lowercase only 
state_FIPS['name_state'] = state_FIPS['name_state'].apply(lambda x: x.lower())

################################
## Load and format demograhpy ##
################################

# load demographic data
state_demo = pd.read_csv(os.path.join(os.getcwd(), '../../raw/demography/sc-est2023-agesex-civ.csv'))
# select right sex and year
state_demo = state_demo[((state_demo['SEX'] == 0) & (state_demo['AGE'] != 999))][['DIVISION','STATE', 'AGE', 'POPEST2020_CIV']]
# aggregate data to total population
agg = state_demo.groupby(['DIVISION','STATE'], observed=False)['POPEST2020_CIV'].sum().reset_index()
# attach region names
region_mapping = ['United States Total', 'New England', 'Middle Atlantic', 'East North Central', 'West North Central', 'South Atlantic', 'East South Central', 'West South Central', 'Mountain', 'Pacific']
# map REGION values to their corresponding names
agg['region_name'] = agg['DIVISION'].map(lambda x: region_mapping[x])

###########################
## Merge and save result ##
###########################

# merge the dataframes on the corresponding columns
merged_df = state_FIPS.merge(agg, left_on="fips_state", right_on="STATE", how="left")
# drop the redundant STATE column after merging
out = merged_df.drop(columns=["STATE", "DIVISION"])
# rename column
out = out.rename(columns={"POPEST2020_CIV": "population"})

# save
out.to_csv(os.path.join(os.getcwd(),'../../interim/demography/demography.csv'), index=False)
