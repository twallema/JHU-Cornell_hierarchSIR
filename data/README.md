# Data readme

Contains an overview of the raw data sources, and the conversion scripts used to convert raw into interim data.

## Raw

### Demography

+ `national_state2020.txt`: Contains the 2020 US state names, abbreviation and corresponding two-digit FIPS. Downloaded from https://www.census.gov/library/reference/code-lists/ansi.html

### Cases

+ `influenza-surveillance-summary_NC_xx-xx.pdf`: End-of-season report on the xx-xx Influenza season in North Carolina. Of special interest is the figure at the bottom of page 6 titled 'Influenza Positive Tests Reported by PHE Facilities', whose data will be extracted using WebPlotDigitizer. Downloaded from https://flu.ncdhhs.gov/data.htm

+ `hosp-admissions_NC_2010-2025.csv`: Weekly hospital admissions in Emergency Departments in North Carolina from 2010 to 2025. Indexed on Saturday. Downloaded from: https://ncdetect.org/respiratory-dashboard/ > Admissions. Alternatively: https://covid19.ncdhhs.gov/dashboard/data-behind-dashboards > Hospital Admissions from the Emergency Department. Made available here: https://github.com/ACCIDDA/NC_Forecasting_Collab/blob/main/nc_data/cleaned/20250116_hosp_admissions.csv by Matthew Mietchen.

+ `ED-visits_NC_2010-2025.csv`: Weekly visits to Emergency Departments in North Carolina for Influenza-like illness (ILI) from 2010 to 2025. Indexed on Saturday. Downloaded from: https://ncdetect.org/respiratory-dashboard/ > Overall Trends. Made available here: https://github.com/ACCIDDA/NC_Forecasting_Collab/blob/main/nc_data/cleaned/20250116_hosp_admissions.csv by Matthew Mietchen.

+ `ICL_NREVSS_Combined_prior_to_2015_16.csv`: Downloaded from FluVIEW interactive https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html. Removed column "A (Unable to Subtype)" as it's almost all zeros and not included in `ICL_NREVSS_Public_Health_Labs.csv`, in order to avoid confusion.

+ `ICL_NREVSS_Public_Health_Labs.csv`: Downloaded from FluVIEW interactive https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html. 

### Vaccination

+ `vacc_alldoses_age_Flu_2024_R1_allflu_allseasons.csv`: Contains, from the 2010-2011 to the 2023-2024 season, the weekly number of administered vaccines ('vacc_age'), per age group ('age_group'), per US state ('subpop'). Obtained from Shaun Truelove.

## Interim

### Demography

+ `demography.csv`: Population of US states. Columns: 'abbreviation_state', 'fips_state', 'name_state', 'population', 'region_name'.

### Cases

+ `subtypes_NC_14-25.csv`: Number of positive tests reported by PHE facilities attributed to Influenza A versus Influenza B Influenza. Data for 2014-2024 were extracted from the end-of-season reports available in the raw data folder: `influenza-surveillance-summary_NC_xx-xx.pdf`. Data for 2025 were downloaded from https://covid19.ncdhhs.gov/dashboard/respiratory-virus-surveillance. The seasons 2012-2013 and 2013-2014 contain the reported season's cumulative totals, repeated from Oct-May.

+ `subtypes_FluVIEW-interactive_14-25.csv`: Contains the weekly subtype interformation for the US' HHS regions from 2014-2025. Columns: "REGION TYPE", "YEAR", "WEEK", "A (H1)", "A(H3)", "B".  Built by combining `ICL_NREVSS_Combined_prior_to_2015_16.csv` and `ICL_NREVSS_Public_Health_Labs.csv`. From `ICL_NREVSS_Public_Health_Labs.csv`, only the column "B" was retained and not the subtyping of B into Yam and Vic, this was not done because this dataset is only used to determine the ratio of A(H1) vs A(H3).  Week 53 of 2015 had to be inserted manually, it is a copy of week 52. Week 53 of 2014 had to be removed, for some weird reason.

### Vaccination
 
+ `vaccination_incidences_2010-2024.csv`: Formats the raw vaccination data from 2010-2024, `vacc_alldoses_age_Flu_2024_R1_allflu_allseasons.csv`, to use the naming conventions used in this software.

### Calibration

+ `baselineModels-accuracy.csv`: Contains the WIS of the GRW baseline model with or without drift, used to normalise our model's WIS scores. Columns: 'model', 'season', 'reference_date', 'horizon', 'WIS'. Generated using `~/scripts/manuscript/optimize_run-baselineModels.py`.

## Conversion

### Demography

+ `build-demography.py`: Script used to build the US state-level demography.

### Vaccination

+ `build-vaccination.py`: A script that formats the vaccination rates of the 2010-2024 season.
