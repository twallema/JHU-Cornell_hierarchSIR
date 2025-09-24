# Data readme

Contains an overview of the raw data sources, and the conversion scripts used to convert raw into interim data.

## Raw

### Demography

+ `national_state2020.txt`: Contains the 2020 US state names, abbreviation and corresponding two-digit FIPS. Downloaded from https://www.census.gov/library/reference/code-lists/ansi.html

## Interim

### Demography

+ `demography.csv`: Population of US states. Columns: 'name_state', 'fips_state', 'name_state', 'population', 'region_name'.

### NHSN-HRD_archive

+ `NHSN-HRD_ending-YYYYWW_gathered-YYYY-MM-DD-HH-MM-SS.parquet`: Hospital incidence for COVID-19, Influenza and RSV in every US state and per epiweek, as obtained from the NHSN HRD dataset. Automatically updated and archived weekly through a Github action: `~/.github/workflows/automated_data_collection.yml`. Columns: 'season', 'year', 'MMWR', 'date', 'fips_state', 'name_state', 'influenza admissions', 'covid-19 admissions', 'rsv admissions'.

### Calibration

To be filled out later.

## Conversion

### Demography

+ `build-demography.py`: Script used to build the US state-level demography.

### Cases

+ `fetch-format_NHSN-HRD-data.py`: Script to collect NHSN HRD data, format it and archive it. Called by through a Github action: `~/.github/workflows/automated_data_collection.yml`.