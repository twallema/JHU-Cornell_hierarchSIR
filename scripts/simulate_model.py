import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from hierarchSIR.model import SIR

# Parameters
initial_condition = {
    'S0': [7E6, 7E6],
    'I0': [100, 1],
    'R0': [3E6, 3E6],
    'I_inc0': [0, 0],
    'H_inc_star0': [0, 0],
    'H_inc0': [0, 0],
    }

parameters = {
    'beta_0': [0.5, 0.5],
    'gamma': [1/3.5, 1/3.5],
    'rho_i': [0.025, 0.025],
    'rho_h': [0.0025, 0.0025],
    'delta_beta_temporal': [1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5],
    'modifier_length': 15,
    'sigma': 2.5
    }

t_start = datetime(2023, 10, 1)
t_stop = datetime(2024, 5, 1)

# Set up model
model = SIR(parameters, initial_condition)

# Wrapper for timer
import timeit
def to_time():
    model.sim(t_start, t_stop)
# Measure the execution time for 20 repetitions
execution_time = timeit.timeit(to_time, number=20)
print(f"Total execution time for 20 runs: {execution_time:.6f} seconds")
print(f"Average execution time per run: {execution_time / 20:.6f} seconds")

# Call C++ function
simout = model.sim(t_start, t_stop)

# Plot results
fig,ax=plt.subplots(nrows=2, ncols=1, figsize=(8.3,11.7/2), sharex=True)
## population dynamics
t = simout.time
ax[0].plot(t, simout['S'].sel(strain=0), label="Susceptible")
ax[0].plot(t, simout['I'].sel(strain=0), label="Infected")
ax[0].plot(t, simout['R'].sel(strain=0), label="Recovered")
## incidence
ax[1].plot(t, simout['I_inc'].sel(strain=0), label="Infected (inc)")
ax[1].plot(t, simout['I_inc'].sel(strain=1), label="Infected (inc), strain 2")
ax[1].plot(t, simout['H_inc'].sel(strain=0), label="Hospitalised (inc)")
ax[1].plot(t, simout['H_inc'].sel(strain=1), label="Hospitalised (inc), strain 2")

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.show()
