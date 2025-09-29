"""
This script downloads, formats and archives the NHSN HRD dataset
"""

__author__      = "T.W. Alleman & Clif McKee"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group (JHUBSPH) & Bento Lab (Cornell CVM). All Rights Reserved."

##################
## Dependencies ##
##################

import os
import argparse
import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta
from hierarchSIR.utils import str_to_bool

# Define relevant global  variables
abs_dir = os.path.dirname(__file__)
collection_datetime_str = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

#####################
## Parse arguments ##
#####################

# preliminary or consolidated dataset?
parser = argparse.ArgumentParser()
parser.add_argument("--preliminary", type=str_to_bool, default=False, help="Download preliminary dataset (available Wednesday) or consolidated dataset (available Saturday).")
args = parser.parse_args()

# change download link and save folder
if args.preliminary:
    url = 'https://data.cdc.gov/api/views/mpgq-jmmr/rows.csv?accessType=DOWNLOAD'
    save_folder = '../../interim/cases/NHSN-HRD_archive/preliminary/'
else:
    url = 'https://data.cdc.gov/api/views/ua7e-t2fy/rows.csv?accessType=DOWNLOAD'
    save_folder = '../../interim/cases/NHSN-HRD_archive/consolidated/'

####################
## Main Functions ##
####################

def get_raw_HRD_data(url: str) -> pd.DataFrame:
    """
    Downloads the Hospital Respiratory Data (HRD) from the National Healthcare Safety Network (NHSN)

    input
    -----

    url: str
        Web location of NHSN HRD dataset

    output
    ------

    raw_HRD_data: pd.DataFrame
        An unaltered "raw" copy of the HRD dataset.

    notes
    -----

    Data source: https://healthdata.gov/dataset/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/n3kj-exp9/about_data > Access this data > TEXT/CSV
    """
    
    return pd.read_csv(url, index_col=0, parse_dates=True).reset_index()

def format_raw_HRD_data(raw_HRD_data: pd.DataFrame) -> pd.DataFrame:
    """
    A function converting the raw Hospital Respiratory Data (HRD) into a more enjoyable interim format

    input
    -----

    raw_HRD_data: pd.DataFrame
        An unaltered "raw" copy of the HRD dataset, obtained using `get_raw_HRD_data()`.

    output
    ------

    interim_HRD_data: pd.DataFrame
        An altered "interim" copy of the HRD dataset. Retains only relevant columns, contains fips codes & region names and is sorted.
        - Retained columns: 'Week Ending Data', 'Geographic aggregation', 'Total COVID-19 Admissions', 'Total Influenza Admissions' and 'Total RSV Admissions'
        - Names changed into: 'date', 'fips_state', 'name_state', 'covid-19 admissions', 'influenza admissions' and 'rsv admissions'
    """

    # Retain only relevant columns
    data = raw_HRD_data[['Week Ending Date', 'Geographic aggregation', 'Total COVID-19 Admissions', 'Total Influenza Admissions', 'Total RSV Admissions']]

    # Get fips mappings
    fips_mappings = pd.read_csv(os.path.join(abs_dir, '../../interim/demography/demography.csv'), dtype={'fips_state': str})

    # Add state FIPS code to dataframe
    state_fips_mapping = fips_mappings[["abbreviation_state", "fips_state"]].drop_duplicates()              # get abbreviation / fips
    mapping_dict = dict(zip(state_fips_mapping["abbreviation_state"], state_fips_mapping["fips_state"]))    # build map
    data["fips_state"] = data["Geographic aggregation"].map(mapping_dict)                                   # append fips codes
    data = data.rename(columns={'Geographic aggregation': 'name_state'})                                    # give better names
    data = data[data['name_state'].isin(state_fips_mapping['abbreviation_state'])]                          # retain only continental US, Alaska, Hawaii and Puerto Rico
 
    # Add a year, epiweek & season label for easy splitting of the dataset later down the line
    data[['year', 'MMWR']] = data['Week Ending Date'].apply(lambda x: pd.Series(get_epiweek(x)))
    data["season"] = data.apply(attach_flu_season_label, axis=1)

    # Give columns more "code-friendly" names
    data = data.rename(columns={'Week Ending Date': 'date', 'Total COVID-19 Admissions': 'covid-19 admissions', 'Total Influenza Admissions': 'influenza admissions', 'Total RSV Admissions': 'rsv admissions'})

    # Omit all seasons pre-2022-2023 (COVID-19 pandemic spoils trends)
    data = data[data['date'] > datetime(2022,9,4)]

    # Re-arrange columns in a logical order
    interim_HRD_data = data[['season', 'year', 'MMWR', 'date', 'fips_state', 'name_state', 'influenza admissions', 'covid-19 admissions', 'rsv admissions']]

    # Perform a sorting step for ease of interpretation
    interim_HRD_data['fips_state'] = pd.to_numeric(interim_HRD_data['fips_state'])
    interim_HRD_data = interim_HRD_data.sort_values(by=["date", "fips_state"], ascending=[True, True])

    return interim_HRD_data


def save_interim_data(interim_data: pd.DataFrame, save_data_path: str) -> None:
    """
    Save a compressed copy of the interim NHSN HRD dataset
    """

    # Determine year + MMWR of last available datapoint
    endyear = interim_data['year'].unique().max()
    endMMWR = interim_data[interim_data['year'] == endyear]['MMWR'].unique().max()

    # Save a copy in the data/interim/cases/collection/NHSN-HRD_archive
    ## Make folder
    desired_path = os.path.join(abs_dir, save_data_path)
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    ## Dump a copy of the raw data
    interim_data.to_parquet(os.path.join(desired_path,f'NHSN-HRD_ending-{endyear}{endMMWR}_gathered-{collection_datetime_str}.parquet.gzip'), compression='gzip', index=False)
    pass


######################
## Helper Functions ##
######################


def get_epiweek(date: datetime) -> Tuple[str, str]:
    """Convert a date string ("YYYY-MM-DD") to CDC epiweek year and week.
    """

    # Compute Thursday of the same week
    days_to_thursday = (3 - date.weekday()) % 7
    thursday = date + timedelta(days=days_to_thursday)

    # Get the epidemiological year (based on Thursday's year)
    epi_year = thursday.year

    # Compute the first Thursday of the epidemiological year
    first_thursday = datetime(epi_year, 1, 4) + timedelta(days=(3 - datetime(epi_year, 1, 4).weekday()) % 7)

    # Compute the MMWR week number (difference in weeks from the first Thursday)
    epi_week = ((thursday - first_thursday).days // 7) + 1

    return epi_year, epi_week


def attach_flu_season_label(row):
    """
    We define an influenza season as epiweek 36 of the current year until epiweek 35 of the next year.
    """
    if row["MMWR"] >= 36:
        return f"{row['year']}-{row['year']+1}"
    else:
        return f"{row['year']-1}-{row['year']}"


######################
## Trigger workflow ##
######################

df_raw = get_raw_HRD_data(url)
df_interim = format_raw_HRD_data(df_raw)
save_interim_data(df_interim, save_folder)