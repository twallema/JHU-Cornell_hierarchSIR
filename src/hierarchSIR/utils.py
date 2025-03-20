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
        'f_I': np.array([1e-4, 1e-5]),
        'f_R': np.array([0.35, 0.35]), 
        # SIR parameters
        'beta_0': [0.5, 0.5],
        'gamma': [1/3.5, 1/3.5],
        # modifiers
        'delta_beta_temporal': np.array([1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5]),
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
        'delta_beta_temporal': np.array([1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5]),
        'modifier_length': 15,
        'sigma': 2.5,
        # observation parameters
        'rho_i': [0.025,],
        'rho_h': [0.0025,],
        'T_h': 3.5
        }

    return SIR(parameters)