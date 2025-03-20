import os
import numpy as np
import pandas as pd
from hierarchSIR.model import SIR

def initialise_model(strains=False):
    """
    A function to intialise the model
    """

    if strains == True:
        # Parameters
        parameters = {
        # initial condition function
        'f_I': np.array([1e-4, 1e-6]),
        'f_R': np.array([0.35, 0.35]), 
        # SIR parameters
        'beta': [0.5, 0.5],
        'gamma': [1/3.5, 1/3.5],
        # modifiers
        'delta_beta_temporal': np.array([1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5])-1,
        'modifier_length': 15,
        'sigma': 2.5,
        # observation parameters
        'rho_i': [0.025, 0.025],
        'rho_h': [0.0025, 0.0025],
        'T_h': 3.5
        }
    else:
        # Parameters
        parameters = {
        # initial condition function
        'f_I': np.array([1e-4,]),
        'f_R': np.array([0.35,]), 
        # SIR parameters
        'beta': [0.5,],
        'gamma': [1/3.5,],
        # modifiers
        'delta_beta_temporal': np.array([1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5])-1,
        'modifier_length': 15,
        'sigma': 2.5,
        # observation parameters
        'rho_i': [0.025,],
        'rho_h': [0.0025,],
        'T_h': 3.5
        }

    return SIR(parameters)

def get_NC_influenza_data(startdate, enddate, season):
    """
    Get the North Carolina Influenza dataset -- containing ED visits, ED admissions and subtype information -- for a given season

    input
    -----

    - startdate: str/datetime
        - start of dataset
    
    - enddate: str/datetime
        - end of dataset

    - season: str
        - influenza season

    output
    ------

    - data: pd.DataFrame
        - index: 'date' [datetime], columns: 'H_inc', 'I_inc', 'H_inc_A', 'H_inc_B' (frequency: weekly, converted to daily)
    """

    # load raw Hospitalisation and ILI data + convert to daily incidence
    data_raw = [
        pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/raw/cases/hosp-admissions_NC_10-25.csv'), index_col=0, parse_dates=True)[['flu_hosp']].squeeze()/7,  # hosp
        pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/raw/cases/ED-visits_NC_10-25.csv'), index_col=0, parse_dates=True)[['flu_ED']].squeeze()/7               # ILI
            ]   
    # rename 
    data_raw[0] = data_raw[0].rename('H_inc')
    data_raw[1] = data_raw[1].rename('I_inc')
    # merge
    data_raw = pd.concat(data_raw, axis=1)
    # change index name
    data_raw.index.name = 'date'
    # slice right dates
    data_raw = data_raw.loc[slice(startdate,enddate)]
    # load subtype data flu A vs. flu B
    df_subtype = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/interim/cases/subtypes_NC_14-25.csv'), index_col=1, parse_dates=True)
    # load right season
    df_subtype = df_subtype[df_subtype['season']==season][['flu_A', 'flu_B']]
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
    return df_merged[['H_inc', 'I_inc', 'H_inc_A', 'H_inc_B']].loc[slice(startdate,enddate)]
