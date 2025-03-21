"""
This script contains a wrapper to simulate the C++ SIR model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import copy
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

# import C++ model
from hierarchSIR import sir_model

# define integration function
class SIR():
    """
    SIR Influenza model
    """
    
    def __init__(self, parameters):
        """
        # TODO: docstring
        """
        # determine the number of strains
        self.n_strains = len(parameters['beta'])
        self.states = ['S', 'I', 'R', 'I_inc', 'H_inc_LCT0', 'H_inc_LCT1', 'H_inc_LCT2', 'H_inc']

        # TODO: retrieve the state's demography
        # add state name or fips as input argument
        self.demography = 11.05E6

        # assign variables to object
        self.parameters = parameters

        pass

    def sim(self, start_date, stop_date, N=1, draw_function=None, draw_function_kwargs={}):
        """
        # TODO: docstring
        """

        # translate start and stop relative to mid Nov
        time = self.convert_dates_to_timesteps(start_date, stop_date)

        # save a copy before altering to reset after simulation
        cp_pars = copy.deepcopy(self.parameters)
        # loop over number of repeated samples
        output = []
        for _ in range(N):
            # get parameters
            if draw_function:
                self.parameters.update(draw_function(copy.deepcopy(self.parameters), **draw_function_kwargs))
            # build initial condition
            initial_condition = self.initial_condition_function(self.demography, self.parameters['f_I'], self.parameters['f_R'])
            # remove ICF arguments from the parameters
            self.parameters = {key: value for key, value in self.parameters.items() if key not in ['f_I', 'f_R']}
            # simulate model
            simout = sir_model.integrate(*time, **initial_condition, **self.parameters)
            # format and append output
            output.append(self.format_output(np.array(simout), start_date, self.states, self.n_strains))
            # Reset parameter dictionary
            self.parameters = cp_pars

        # Concatenate along dimension 'draw'
        out = output[0]
        for xarr in output[1:]:
            out = xr.concat([out, xarr], "draws")

        return out

    @staticmethod
    def initial_condition_function(demography, f_I, f_R):
        """
        A function generating the model's initial condition.
        
        input
        -----

        demography: float
            Number of inhabitants in US state

        f_I: float
            Fraction of the population initially infected
        
        f_R: float
            Fraction of the population initially immune

        output
        ------

        initial_condition: dict
            Keys: 'S0', ... . Values: np.ndarray.
        """

        # construct initial condition
        return {'S0':  (1 - f_I - f_R) * demography,
                'I0': f_I * demography,   
                'R0': f_R * demography,
                }

    @staticmethod
    def format_output(simout: np.ndarray, start_date: datetime, states: list, n_strains: int) -> xr.Dataset:
        """
        Convert the C++ SIR model output to an xarray.Dataset.

        Parameters:
        -----------

        - simout (np.ndarray): The output of `sir_model.integrate()`, shape (n_timesteps, 1 + n_states * n_strains).
        - start_time (datetime): The start date of the simulation.
        - states (list): The names of the model's states.
        - n_strains (int): Number of influenza strains.

        Returns:
        --------

        - xr.Dataset: A structured xarray dataset with labeled dimensions (date, strain).
        """

        # Extract simulation time and state variables
        date = pd.to_datetime([datetime(start_date.year, 10, 15) + timedelta(days=t) for t in simout[:, 0]])
        n_timesteps = simout.shape[0]
        n_states = len(states)

        # Reshape data from (n_timesteps, n_states * n_strains) -> (n_timesteps, n_states, n_strains)
        data_values = simout[:, 1:].reshape(n_timesteps, n_strains, n_states).swapaxes(1, 2)

        # Dynamically construct the data_vars dictionary using state names
        data_vars = {state: (["date", "strain"], data_values[:, i, :]) for i, state in enumerate(states)}

        # Create xarray dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "date": date,
                "strain": np.arange(n_strains),
            },
        )

        return ds
    

    @staticmethod
    def convert_dates_to_timesteps(start_date: datetime, end_date: datetime) -> list:
        """
        Convert absolute start and end datetimes into integer indices relative to October 15 of the same year.

        Parameters:
        - start_date (datetime): The start date of the simulation.
        - end_date (datetime): The end date of the simulation.

        Returns:
        - (int, int): The start and end dates as integer indices relative to the C++ model's t=0 of October 15.
        """

        # Define reference date (Oct 15 of the start_date's year)
        reference_date = datetime(start_date.year, 10, 15)

        # Compute integer days relative to reference date
        start_index = (start_date - reference_date).days
        end_index = (end_date - reference_date).days

        return [start_index, end_index]
