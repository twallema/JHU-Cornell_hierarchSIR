"""
This script contains the Python wrapper to the C++ SIR model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group (JHUBSPH) & Bento Lab (Cornell CVM). All Rights Reserved."

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
class imsSIR():
    """
    Independent Multi-strain SIR model
    """
    
    def __init__(self, parameters, initial_condition_function, n_strains):
        """
        # TODO: docstring
        """

        # assign variables to object
        self.parameters = parameters
        self.n_strains = n_strains

        # take in the initial condition function and retrieve its arguments
        self.ICF = initial_condition_function
        self.ICF_args_names = list(inspect.signature(initial_condition_function).parameters.keys())
        self.initial_condition = self.ICF(*[self.parameters[par] for par in self.ICF_args_names])

        # attributes needed to make model compatible with pySODM's optimization module
        self.states_names = ['S', 'I', 'R', 'I_inc', 'H_inc']
        self.coordinates = {'strain': np.linspace(start=0, stop=n_strains-1, num=n_strains).tolist()}
        self.state_shapes, self.dimensions_per_state, self.state_coordinates = build_state_sizes_dimensions(self.coordinates, self.states_names, None)
        self.initial_states = {'S': np.zeros(n_strains),  'I': np.zeros(n_strains), 'R': np.zeros(n_strains), 'I_inc': np.zeros(n_strains), 'H_inc': np.zeros(n_strains)}
        _, self.parameter_shapes = validate_calibrated_parameters([par for par in self.parameters.keys() if par != 'season'], self.parameters)

        pass

    def sim(self, simtime, atol=1e-2, rtol=1e-5, N=1, draw_function=None, draw_function_kwargs={}):
        """
        Simulate the model

        Parameters:
        -----------

        - simtime: list containing datetime
            - start and enddate of the simulation
        
        - atol: float
            - absolute tolerance of the runge_kutta_dopri5 stepper 
        
        - rtol: float
            - relative tolerance of the runge_kutta_dopri5 stepper 
        
        - N: int
            - number of repeated simulations
        
        - draw_function: callable
            - function describing a sampling of model parameters
        
        - draw_function_kwargs: dict
            - arguments of `draw_function`

        Returns:
        --------

        - simout: xarray.Dataset
            - simulation output
        """

        # translate start and stop relative to reference date + check chronology
        start_date, stop_date = simtime
        assert start_date < stop_date, 'start date of simulations must fall before end date'
        # define reference date (date at which t=0 in C++ model --> = date at which modifiers start)
        reference_date = datetime(start_date.year, 10, 15)
        # translate input to C++ model time indices
        time = self.dates_to_simtime(start_date, stop_date, reference_date)

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
            self.initial_condition = self.ICF(*[self.parameters[par] for par in self.ICF_args_names])
            # remove ICF arguments from the parameters
            self.parameters = {key: value for key, value in self.parameters.items() if key not in self.ICF_args_names}
            # simulate model
            simout = sir_model.integrate(*time, atol, rtol, **self.initial_condition, **self.parameters)
            # format and append output
            output.append(self.format_output(np.array(simout), reference_date, self.states_names, self.n_strains))
            # Reset parameter dictionary
            self.parameters = cp_pars

        # Concatenate along dimension 'draw'
        out = output[0]
        for xarr in output[1:]:
            out = xr.concat([out, xarr], "draws")

        return out

    @staticmethod
    def format_output(simout: np.ndarray, reference_date: datetime, states: list, n_strains: int) -> xr.Dataset:
        """
        Convert the C++ SIR model output to an xarray.Dataset.

        Parameters:
        -----------

        - simout (np.ndarray): The output of `sir_model.integrate()`, shape (n_timesteps, 1 + n_states * n_strains).
        - reference_date (datetime): The date at which the C++ model's internal timekeeping is equal to t=0.
        - states (list): The names of the model's states.
        - n_strains (int): Number of influenza strains.

        Returns:
        --------

        - xr.Dataset: A structured xarray dataset with labeled dimensions (date, strain).
        """

        # Extract simulation time and state variables
        date = pd.to_datetime([reference_date + timedelta(days=t) for t in simout[:, 0]])
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
    def dates_to_simtime(start_date: datetime, end_date: datetime, reference_date: datetime) -> list:
        """
        Convert start and end datetimes of the simulation to a list of integer timesteps, expressed relative to a reference_date.

        Parameters:
        -----------

        - start_date (datetime): The start date of the simulation.
        - end_date (datetime): The end date of the simulation.
        - reference_date (datetime): The date at which the C++ model's internal timekeeping is equal to t=0.

        Returns:
        --------

        - (int, int): The start and end dates of the simulation translated to integer indices representing the number of days difference with the reference date
        """

        return [(start_date - reference_date).days, (end_date - reference_date).days]
