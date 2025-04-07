# Data readme

Contains an overview of the raw data sources, and the conversion scripts used to convert raw into interim data.

## Raw

### Demography

+ `national_state2020.txt`: Contains the 2020 US state names and corresponding two-digit FIPS. Downloaded from https://www.census.gov/library/reference/code-lists/ansi.html

+ `sc-est2023-agesex-civ.csv`: Contains the estimated population per year of age, sex and US state. Downloaded from https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-detail.html 

### cases

+ `influenza-surveillance-summary_NC_xx-xx.pdf`: End-of-season report on the xx-xx Influenza season in North Carolina. Of special interest is the figure at the bottom of page 6 titled 'Influenza Positive Tests Reported by PHE Facilities', whose data will be extracted using WebPlotDigitizer. Downloaded from https://flu.ncdhhs.gov/data.htm

+ `hosp-admissions_NC_2010-2025.csv`: Weekly hospital admissions in Emergency Departments in North Carolina from 2010 to 2025. Indexed on Saturday. Downloaded from: https://ncdetect.org/respiratory-dashboard/ > Admissions. Alternatively: https://covid19.ncdhhs.gov/dashboard/data-behind-dashboards > Hospital Admissions from the Emergency Department. Made available here: https://github.com/ACCIDDA/NC_Forecasting_Collab/blob/main/nc_data/cleaned/20250116_hosp_admissions.csv by Matthew Mietchen.

+ `ED-visits_NC_2010-2025.csv`: Weekly visits to Emergency Departments in North Carolina for Influenza-like illness (ILI) from 2010 to 2025. Indexed on Saturday. Downloaded from: https://ncdetect.org/respiratory-dashboard/ > Overall Trends. Made available here: https://github.com/ACCIDDA/NC_Forecasting_Collab/blob/main/nc_data/cleaned/20250116_hosp_admissions.csv by Matthew Mietchen.

+ `ICL_NREVSS_Combined_prior_to_2015_16.csv`: Downloaded from FluVIEW interactive https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html. 

+ `ICL_NREVSS_Public_Health_Labs.csv`: Downloaded from FluVIEW interactive https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html. Removed the first row. 

## Interim

### Demography

+ `demography.csv`: Population of US states. Columns: 'abbreviation_state', 'fips_state', 'name_state', 'population', 'region_name'.

### Cases

+ `subtypes_NC_14-25.csv`: Number of positive tests reported by PHE facilities attributed to Influenza A versus Influenza B Influenza. Data for 2014-2024 were extracted from the end-of-season reports available in the raw data folder: `influenza-surveillance-summary_NC_xx-xx.pdf`. Data for 2025 were downloaded from https://covid19.ncdhhs.gov/dashboard/respiratory-virus-surveillance. The seasons 2012-2013 and 2013-2014 contain the reported season's cumulative totals, repeated from Oct-May.

+ `subtypes_FluVIEW-interactive_14-25.csv`: Contains the weekly subtype interformation for the US' HHS regions from 2014-2025. Columns: "REGION TYPE", "YEAR", "WEEK", "A (H1)", "A(H3)", "B".  Built by combining `ICL_NREVSS_Combined_prior_to_2015_16.csv` and `ICL_NREVSS_Public_Health_Labs.csv`.

### Calibration

## Conversion

### Demography

+ `build-demography.py`: Script used to build the US state-level demography.