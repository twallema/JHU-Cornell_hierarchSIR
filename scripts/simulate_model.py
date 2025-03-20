import numpy as np
import sir_model # Import compiled C++ module
import matplotlib.pyplot as plt

# Parameters
S0 = [7E6, 7E6]
I0 = [100, 1]
R0 = [3E6, 3E6]
I_inc0 = [0, 0]
beta_0 = [0.5, 0.5]
gamma = [1/3.5, 1/3.5]
t0 = -30
t_end = 210
dt = 1

modifier_length = 15  # Each modifier applies for 15 days
sigma = 2.5  # Gaussian filter width (adjust for more or less smoothing)

# Time-varying beta values
delta_beta_temporal = [1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5]

# Wrapper for timer
import timeit
def to_time():
    sir_model.sir_model(S0, I0, R0, I_inc0, beta_0, gamma, delta_beta_temporal, modifier_length, sigma, t0, t_end, dt)
# Measure the execution time for 20 repetitions
execution_time = timeit.timeit(to_time, number=20)
print(f"Total execution time for 20 runs: {execution_time:.6f} seconds")
print(f"Average execution time per run: {execution_time / 20:.6f} seconds")

# Call C++ function
results = sir_model.sir_model(S0, I0, R0, I_inc0, beta_0, gamma, delta_beta_temporal, modifier_length, sigma, t0, t_end, dt)

# Convert to NumPy array
results = np.array(results)

# Plot results
fig,ax=plt.subplots(nrows=2, ncols=1, figsize=(8.3,11.7/2), sharex=True)
## population dynamics
t = results[:, 0]
ax[0].plot(t, results[:, 1], label="Susceptible")
ax[0].plot(t, results[:, 2], label="Infected")
ax[0].plot(t, results[:, 3], label="Recovered")
## incidence
ax[1].plot(t, results[:, 2], label="Infected (prev)")
ax[1].plot(t, results[:, 4], label="Infected (inc)")
ax[1].plot(t, results[:, 6], label="Infected (prev), strain 2")
ax[1].plot(t, results[:, 8], label="Infected (inc), strain 2")


plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.title(f"SIR Model with Smoothed Beta Modifiers (σ={sigma})")
plt.show()
