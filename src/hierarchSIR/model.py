"""
This script contains a wrapper to simulate the C++ SIR model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import copy
import inspect
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

# import C++ model
from hierarchSIR import sir_model

# import pySODM modules
from pySODM.models.validation import build_state_sizes_dimensions
from pySODM.optimization.utils import validate_calibrated_parameters

# define integration function
class SIR():
    """
    SIR Influenza model
    """
    
    def __init__(self, parameters, initial_condition_function, n_strains):
        """
        # TODO: docstring
        """

        # take in the initial condition function and retrieve its arguments
        self.ICF = initial_condition_function
        self.ICF_args_names = list(inspect.signature(initial_condition_function).parameters.keys())

        # assign variables to object
        self.parameters = parameters
        self.n_strains = n_strains

        # attributes needed to make model compatible with pySODM's optimization module
        self.states_names = ['S', 'I', 'R', 'I_inc', 'H_inc']
        self.coordinates = {'strain': np.linspace(start=0, stop=n_strains-1, num=n_strains).tolist()}
        self.state_shapes, self.dimensions_per_state, self.state_coordinates = build_state_sizes_dimensions(self.coordinates, self.states_names, None)
        self.initial_states = {'S': np.zeros(n_strains),  'I': np.zeros(n_strains), 'R': np.zeros(n_strains), 'I_inc': np.zeros(n_strains), 'H_inc': np.zeros(n_strains)}
        _, self.parameter_shapes = validate_calibrated_parameters(self.parameters.keys(), self.parameters)

        pass

    def sim(self, simtime, N=1, draw_function=None, draw_function_kwargs={}):
        """
        # TODO: docstring
        """

        # translate start and stop relative to mid Nov
        start_date, stop_date = simtime
        time = self.convert_dates_to_timesteps(start_date, stop_date)

        # save a copy before altering to reset after simulation
        cp_pars = copy.deepcopy(self.parameters)
        # loop over number of repeated samples
        output = []
        for _ in range(N):
            # get parameters
            if draw_function:
                self.parameters.update(draw_function(copy.deepcopy(self.parameters), **draw_function_kwargs))
            # make sure parameters are vectors #TODO: do better!
            for par, shape in self.parameter_shapes.items():
                if ((shape == (1,)) & (par not in ['T_h', 'gamma', 'sigma', 'modifier_length']) & (par != 'delta_beta_temporal')):
                    self.parameters[par] = np.array([self.parameters[par],])
            # build initial condition
            initial_condition = self.ICF(*[self.parameters[par] for par in self.ICF_args_names])
            # remove ICF arguments from the parameters
            self.parameters = {key: value for key, value in self.parameters.items() if key not in ['f_I', 'f_R']}
            # simulate model
            simout = sir_model.integrate(*time, **initial_condition, **self.parameters)
            # format and append output
            output.append(self.format_output(np.array(simout), start_date, self.states_names, self.n_strains))
            # Reset parameter dictionary
            self.parameters = cp_pars

        # Concatenate along dimension 'draw'
        out = output[0]
        for xarr in output[1:]:
            out = xr.concat([out, xarr], "draws")

        return out

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
