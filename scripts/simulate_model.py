import matplotlib.pyplot as plt
from datetime import datetime
from hierarchSIR.utils import initialise_model

# start and end of simulation
t_start = datetime(2023, 10, 1)
t_stop = datetime(2024, 8, 1)

# Set up model
model = initialise_model(strains=True)

# Wrapper for timer
import timeit
def to_time():
    model.sim([t_start, t_stop])
# Measure the execution time for 20 repetitions
execution_time = timeit.timeit(to_time, number=20)
print(f"Total execution time for 20 runs: {execution_time:.6f} seconds")
print(f"Average execution time per run: {execution_time / 20:.6f} seconds")

# Call C++ function
simout = model.sim([t_start, t_stop])

# Plot results
fig,ax=plt.subplots(nrows=2, ncols=1, figsize=(8.3,11.7/2), sharex=True)
## population dynamics
t = simout.date
ax[0].plot(t, simout['S'].sel(strain=0), label="Susceptible")
ax[0].plot(t, simout['I'].sel(strain=0), label="Infected")
ax[0].plot(t, simout['R'].sel(strain=0), label="Recovered")
ax[0].legend()
## incidence
ax[1].plot(t, simout['I_inc'].sel(strain=0), label="I_inc")
ax[1].plot(t, simout['H_inc'].sel(strain=0), label="H_inc")

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.show()
