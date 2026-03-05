# standard python libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# pyMC / pytensor
import pymc as pm
import pymc.sampling.jax
import arviz
import pytensor
import pytensor.tensor as pt
pytensor.config.cxx = '/usr/bin/clang++'
pytensor.config.on_opt_error = "ignore"
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
# jax and diffrax
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
import diffrax
import optax
# hierarchSIR
from hierarchSIR.utils import get_NC_influenza_data

# Get North Carolina dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# convert to a list of start and enddates (datetime)
n_modifiers = 26
modifier_length = 7
population = 11E6
seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2023-2024', '2024-2025']        # script works with only one season
n_seasons = len(seasons)
n_observations = 31
start_calibration_month = 10    # (year X)
end_calibration_month = 5       # (year X+1)
start_calibrations = [datetime(int(season[0:4]), start_calibration_month, 1) for season in seasons]
modifier_reference_dates = [datetime(int(season[0:4]), 10, 15) for season in seasons]
start_simulation = -15 # (October 1)

def get_data(use_ED_visits, start_calibrations, modifier_reference_dates, n_observations):
    """
    A function formatting the model's input data

    output:
    -------
    data --> (n_season, n_variables, n_observations)
    eval_dates --> (n_season, n_observations)```
    """

    eval_dates = []
    data = []
    # loop over seasons
    for start_calibration, modifier_reference_date in zip(start_calibrations, modifier_reference_dates):
        # get the data & trim temporally
        df = get_NC_influenza_data(start_calibration, None).iloc[:n_observations]
        # save the time index (per season)
        eval_dates.append([(d.astype('datetime64[ms]').astype('O') - modifier_reference_date)/timedelta(days=1) for d in df.index.values])
        # arrange data
        data_season = df['H_inc'].values
        # prepend ED visits
        if use_ED_visits:
            data_season.insert(0, df['I_inc'].values)
        # stack data to (n_variables, n_observations)
        data.append(np.stack(data_season))
    # stack data to (n_season, n_variables, n_observations)
    data = np.stack(data, axis=0)
    eval_dates = np.stack(eval_dates, axis=0)

    return data, eval_dates

# get the data
data, ts = get_data(False, start_calibrations, modifier_reference_dates, n_observations) # (n_season, n_variables, n_observations)


# Define a jax-jitted diffrax differential equation model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# jax-compatible gaussian smoother
def smooth_with_gaussian(vec, sigma=2.0):
    window_size = 15
    # Build a Gaussian kernel
    x = jnp.linspace(-3, 3, window_size)
    kern = jnp.exp(-0.5 * (x/sigma)**2)
    kern = kern / kern.sum()
    return convolve(vec, kern, mode="same")

# define delta_beta[t] modifier function
def make_delta_beta_daily(delta_beta, duration, t0, t1, sigma=2.5):
    """
    Parameters
    ----------
    delta_beta : array (K,)
        Modifier values for each block.
    duration : int
        Number of days each entry is repeated.
    t0 : int
        Start day of the simulation (can be negative).
    t1 : int
        End day of the simulation (may exceed total expanded length).
    sigma: float
        Standard deviation of the Gausian filter.

    Returns
    -------
    vec : 2D array, shape: 2 x (t1 - t0)
        First row: Timesteps.
        Second row: Delta_beta(t) series with zero-padding outside support.
    """

    # Total support length = len(delta_beta) * duration
    total_len = delta_beta.shape[0]* duration

    # Indices of simulation range
    ts = jnp.arange(t0, t1)

    # Compute block boundaries
    # block i covers [i*duration , (i+1)*duration - 1]
    block_ids = jnp.floor_divide(ts, duration).astype(jnp.int32)

    # Mask: valid only if in range
    valid = (ts >= 0) & (ts < total_len)

    # Gather values, using mod-safe indexing (will be masked out anyway)
    expanded = jnp.where(valid, delta_beta[block_ids], 0.0)

    # Smooth with a guassian filter
    x = jnp.linspace(-7, 7, num=15)
    kern = jnp.exp(-0.5 * (x/sigma)**2)
    kern = kern / kern.sum()
    expanded = convolve(expanded, kern, mode="same")

    return jnp.stack([ts, expanded])

# define ODE rhs
def SIR_vector_field(t, y, args):
    # unpack states and parameters
    S, I, R, H = y
    beta, delta_beta_daily, gamma, rho_h = args
    # prevent negative state values due to rounding errors
    S = jax.nn.softplus(S)
    I = jax.nn.softplus(I)
    R = jax.nn.softplus(R)
    H = jax.nn.softplus(H)
    # compute total population
    N = S + I + R
    # get modifier
    delta_beta = 1 + jnp.interp(t, xp=delta_beta_daily[0,:], fp=delta_beta_daily[1,:])
    # compute state derivatives
    FOI = delta_beta * beta * I / N
    dS = - S * FOI
    dI = S * FOI - gamma * I
    dR = gamma * I
    # observation
    dH = rho_h * S * FOI - H
    return jnp.array([dS, dI, dR, dH])

# build jax model wrapper
def stop_gradients(x):
    return jax.tree.map(jax.lax.stop_gradient, x)

def sol_op_jax(args_diff, args_nodiff, args_static):
    # unpack differentiable parameters
    beta = args_diff[0]
    rho_h = args_diff[1]
    f_I = args_diff[2]
    f_R = args_diff[3]
    delta_beta = args_diff[4:]
    # unpack non-differentiable parameters and block their gradients
    args_nodiff = stop_gradients(args_nodiff)
    gamma = args_nodiff[0]
    ts = args_nodiff[1:]
    # unpack static arguments
    t0, t1_max, modifier_length, population = args_static
    # evaluate modifiers
    delta_beta_daily = make_delta_beta_daily(delta_beta, modifier_length, t0, t1_max)
    # wrap ODE rhs
    term = diffrax.ODETerm(SIR_vector_field)
    # solve ODE
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=t0,
        t1=t1_max,
        dt0=0.1,
        y0=population * jnp.array([1-f_I-f_R, f_I, f_R, 0]),
        args = (beta, delta_beta_daily, gamma, rho_h),
        saveat=diffrax.SaveAt(ts=list(ts)),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-4)
    )
    return sol.ys[:,-1] # return observed state only


# Vectorized multi-season solver ------------------------------

def sol_op_single(args_diff, args_nodiff, args_static):
    """Wrapper for sol_op_jax to allow vmap."""
    return sol_op_jax(args_diff, args_nodiff, args_static)

# vmap over the first axis of each argument
sol_op_multi = jax.vmap(sol_op_single,
                        in_axes=(0, 0, None),   # each season gets its own slice
                        out_axes=0)          # stack results for all seasons

# jit it
jitted_sol_op_multi = jax.jit(sol_op_multi, static_argnums=2)


# Define jax VJP (gradient computation) function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def vjp_sol_op_multi(args_diff, gz, args_nodiff, args_static):
    # vectorize the single-season VJP
    single_vjp = lambda ad, g, an: jax.vjp(
        lambda th: sol_op_jax(th, an, args_static),
        ad
    )[1](g)[0]  # take only gradient w.r.t args_diff

    # vmap over the season dimension
    return jax.vmap(single_vjp, in_axes=(0,0,0))(args_diff, gz, args_nodiff)

# jit it
jitted_vjp_sol_op_multi = jax.jit(vjp_sol_op_multi, static_argnums=3)


# Define the Op and VJPOp classes for the ODE problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SolOp(Op):
    def __init__(self, args_static):
        self.args_static = args_static

    def make_node(self, args_diff, args_nodiff):
        args_diff = pt.as_tensor_variable(args_diff)
        args_nodiff = pt.as_tensor_variable(args_nodiff)
        return Apply(self, [args_diff, args_nodiff], [pt.matrix()])

    def perform(self, node, inputs, outputs):
        args_diff, args_nodiff = inputs
        ys = jitted_sol_op_multi(args_diff, args_nodiff, self.args_static)
        outputs[0][0] = np.asarray(ys)

    def grad(self, inputs, output_grads):
        args_diff, args_nodiff = inputs
        (gz,) = output_grads

        grad_wrt_args_diff = vjp_sol_op(args_diff, gz, args_nodiff)
        grad_wrt_args_nodiff = pt.zeros_like(args_nodiff)  # block gradients

        return [grad_wrt_args_diff, grad_wrt_args_nodiff]


class VJPSolOp(Op):
    def __init__(self, args_static):
        self.args_static = args_static

    def make_node(self, args_diff, gz, args_nodiff):
        return Apply(self, [
            pt.as_tensor_variable(args_diff),   
            pt.as_tensor_variable(gz),         
            pt.as_tensor_variable(args_nodiff)  
        ], [pt.matrix()])                      

    def perform(self, node, inputs, outputs):
        args_diff, gz, args_nodiff = inputs

        # Use the new batched VJP
        grad = vjp_sol_op_multi(args_diff, gz, args_nodiff, self.args_static)

        # Convert to NumPy array for Theano
        outputs[0][0] = np.asarray(grad)

# Register with jax
# ~~~~~~~~~~~~~~~~~

@jax_funcify.register(SolOp)
def sol_op_jax_funcify(op, **kwargs):
    return lambda args_diff, args_nodiff: jitted_sol_op_multi(args_diff, args_nodiff, op.args_static)

@jax_funcify.register(VJPSolOp)
def vjp_sol_op_jax_funcify(op, **kwargs):
    return lambda args_diff, gz, args_nodiff: jitted_vjp_sol_op_multi(args_diff, gz, args_nodiff, op.args_static)


# Pre-optimize the forward simulation model's parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# args diff initial guesses (ballpark estimates)
beta = 0.455
rho_h = 0.0025
f_I = 1e-4
f_R = 0.25
delta_beta_vals = jnp.zeros(n_modifiers)

# compute gradient-safe transformations
args_diff = jnp.concatenate([
                jnp.array([jnp.log(jnp.exp(beta) - 1), jnp.log(jnp.exp(rho_h) - 1), jnp.log(jnp.exp(f_I) - 1), jnp.log(f_R / (1 - f_R))]),
                jnp.arctanh(delta_beta_vals / 0.25)
            ])
args_diff = jnp.expand_dims(args_diff, 0).repeat(n_seasons, axis=0)

# stack args_nodiff per season
args_nodiff = [jnp.concatenate([jnp.array([1/3.5,]), jnp.array(ts[i,:])]) for i in range(n_seasons)]
args_nodiff = np.array(jnp.stack(args_nodiff))


# static arguments
args_static = (start_simulation, max(ts[:,-1]), modifier_length, population)

# define SSE likelihood
def neg_log_likelihood(args_diff):
    # 1. convert back to untransformed values
    block_1 = jax.nn.softplus(args_diff[:, 0:3])  # beta, rho_h, f_I
    block_2 = jnp.expand_dims(jax.nn.sigmoid(args_diff[:, 3]), axis=1)       # f_R
    block_3 = 0.25 * jnp.tanh(args_diff[:, 4:])    # delta_beta
    # 2. pack blocks into args_diff
    args_diff = jnp.concatenate([block_1, block_2, block_3], axis=1)
    # 3. run simulation
    pred = jitted_sol_op_multi(args_diff, args_nodiff, args_static)
    # 3. compute SSE loss
    return jnp.sum((data - pred)**2)


# optimize
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(args_diff)
for i in range(500):
    loss, grads = jax.value_and_grad(neg_log_likelihood)(args_diff)
    updates, opt_state = optimizer.update(grads, opt_state)
    args_diff = optax.apply_updates(args_diff, updates)
    if i % 100 == 0:
        print(i, float(loss))

# convert back to untransformed values
block_1 = jax.nn.softplus(args_diff[:, 0:3])  # beta, rho_h, f_I
block_2 = jnp.expand_dims(jax.nn.sigmoid(args_diff[:, 3]), axis=1)       # f_R
block_3 = 0.25 * jnp.tanh(args_diff[:, 4:])    # delta_beta
args_diff = jnp.concatenate([block_1, block_2, block_3], axis=1)

# run simulation
out = jitted_sol_op_multi(args_diff, args_nodiff, args_static)

# inspect result
fig,ax=plt.subplots(nrows=n_seasons)
for i in range(n_seasons):
    ax[i].plot(ts[i,:], 7*out[i,:], color='red')
    ax[i].scatter(ts[i,:], 7*data[i,:], marker='o', color='black')
os.makedirs('trace', exist_ok=True)
plt.savefig(f'trace/initial-optim.pdf')
plt.close()

# store 1D vector per variable so we can start the chains easily
beta_opt = np.expand_dims(np.array(args_diff[:,0]), axis=1)
rho_h_opt = np.expand_dims(np.array(args_diff[:,1]), axis=1)
f_I_opt = np.expand_dims(np.array(args_diff[:,2]), axis=1)
f_R_opt = np.expand_dims(np.array(args_diff[:,3]), axis=1)
delta_beta_opt = np.array(args_diff[:,4:])
delta_beta_mu_opt = np.mean(delta_beta_opt, axis=0)
eta_opt = np.transpose(delta_beta_opt - delta_beta_mu_opt[None, :]) / np.sqrt(0.01)
eta_opt = eta_opt[1:,:]

# Build pyMC model
# ~~~~~~~~~~~~~~~~

# Compile forward simulation model
sol_op = SolOp(args_static)
vjp_sol_op = VJPSolOp(args_static)

p = 2

# Build pyMC probablistic model
with pm.Model() as model:

    # Hyperparameters
    rho_h_mu = pm.HalfNormal('rho_h_mu', sigma=1e-2/3)
    rho_h_sigma = pm.HalfNormal('rho_h_sigma', sigma=1/3)
    f_I_mu = pm.HalfNormal('f_I_mu', sigma=1e-3)
    f_I_sigma = pm.HalfNormal('f_I_sigma', sigma=1/3)
    f_R_a = pm.HalfNormal('f_R_a', sigma=5)
    f_R_b = pm.HalfNormal('f_R_b', sigma=5)

    # Differentiable parameters (those we wish to calibrate)
    beta = pt.as_tensor_variable(0.455*np.ones(shape=(n_seasons,1)))
    rho_h = pm.LogNormal("rho_h", mu=np.log(rho_h_mu), sigma=rho_h_sigma, size=(n_seasons, 1))
    f_I = pm.LogNormal("f_I", mu=np.log(f_I_mu), sigma=f_I_sigma, size=(n_seasons, 1))
    f_R = pm.Beta("f_R", alpha=f_R_a, beta=f_R_b, size=(n_seasons, 1))


    # ------- ARMA-GARCH modifiers -------

    # Hyperparameter for delta_beta_temporal
    delta_beta_mu = pm.Normal("delta_beta_mu", mu=0, sigma=0.1, shape=n_modifiers)
    
    # GARCH parameters
    omega = pm.HalfNormal("omega", sigma=0.001) # baseline variance (positive)
    a_garch = pm.Beta("a_garch", alpha=2, beta=2) # sensitivity to shocks
    b_garch = pm.Beta("b_garch", alpha=2, beta=2) # shock retention
    sigma2_0 = pm.HalfNormal("sigma2_0", sigma=0.01, shape=n_seasons) # initial position

    # --- Harmonically decaying AR(p) kernel ---
    # Initial position
    z_0 = pm.Normal("z_0", mu=0.0, sigma=0.01, size=(p, n_seasons)) 
    eps_0 = pm.Normal("eps_0", mu=0.0, sigma=0.01, size=n_seasons)
    # Total AR strength (controls overall magnitude)
    sum_psi = pm.Beta("sum_psi", alpha=5, beta=1)
    # Harmonic kernel: 1 / (k+1)^gamma (larger = faster decay)
    decay_psi = pm.Normal("decay_psi", mu=1, sigma=1/3)
    k_psi = pt.arange(p) + 1
    psi_raw = k_psi ** (-decay_psi)
    # Normalize so sum(psi) = sum_psi
    psi = pm.Deterministic("psi",sum_psi * psi_raw / pt.sum(psi_raw))

    # sample iid standard normals as shocks ----------
    eta = pm.Normal("eta", mu=0.0, sigma=1.0, shape=(n_modifiers-1, n_seasons))

    def step(eta_t, prev_z_stack, prev_sigma2, prev_eps, psi, omega, a_garch, b_garch):
        # --- GARCH(1,1) short-term shocks innovation scale ---
        sigma2 = omega + a_garch * (prev_eps ** 2) + b_garch * prev_sigma2
        eps = eta_t * pt.sqrt(sigma2)
        # --- AR(p) short-term shocks ---
        z = pt.sum(psi[:, None] * prev_z_stack, axis=0) + eps
        # --- shift lag stack ---
        new_z_stack = pt.concatenate([z[None, :], prev_z_stack[:-1]], axis=0)
        return new_z_stack, sigma2, eps
    
    # Provide initial states as a list (must match order of returned states)
    outputs_info = [z_0, sigma2_0, eps_0]

    # Run scan over T steps
    (z_stack_seq, sigma2_seq, eps_seq), _ = pytensor.scan(
        fn=step,
        sequences=[eta,],
        outputs_info=outputs_info,
        non_sequences=[psi, omega, a_garch, b_garch],
    )

    # Prepend the initial states u_0, ell_0, sigma2_0
    z_seq = pt.concatenate([z_0[0][None, :], z_stack_seq[:, 0, :]], axis=0)
    sigma2_seq = pt.concatenate([sigma2_0[None, :], sigma2_seq], axis=0)
    eps_seq = pt.concatenate([eps_0[None, :], eps_seq], axis=0)

    # Register deterministic variables to inspect later
    delta_beta = pm.Deterministic("delta_beta", pt.transpose(z_seq) + delta_beta_mu)
    z = pm.Deterministic("z", pt.transpose(z_seq))
    sigma2 = pm.Deterministic("sigma2", pt.transpose(sigma2_seq))
    eps = pm.Deterministic("eps", pt.transpose(eps_seq))

    # Build forward simulation arguments
    args_diff = pt.concatenate(
        [beta, rho_h, f_I, f_R, delta_beta],
        axis=1
    )

    # Run forward simulation model model
    ys = 7*sol_op(args_diff, args_nodiff)
    ys = pt.math.softplus(ys)

    # Likelihood
    alpha = pm.HalfNormal("alpha", sigma=0.001)
    data = pm.NegativeBinomial("data", mu=ys, alpha=1/alpha, observed=7*data)


# Sample pyMC model
# ~~~~~~~~~~~~~~~~~

with model:
    # NUTS
    trace = pm.sample(150, tune=50, chains=24, init='adapt_full', cores=1, progressbar=True, target_accept=0.8, max_treedepth=8,
                     initvals=24*[{'alpha': 0.01, 'delta_beta_mu': delta_beta_mu_opt, 'rho_h': rho_h_opt, 'f_I': f_I_opt, 'f_R': f_R_opt},])
    # SMC
    #trace = pm.smc.sample_smc(draws=10000, chains=12, cores=12, progressbar=True)
    # DEMetroplisZ
    #trace = pm.sample(100000, tune=100000, chains=1, cores=1, progressbar=True, step=pm.DEMetropolisZ(),
    #                   initvals=1*[{'alpha': 0.01, 'delta_beta_mu': delta_beta_mu_opt, 'rho_h': rho_h_opt, 'f_I': f_I_opt, 'f_R': f_R_opt},])

# Generate traces
variables2plot = [
                'alpha',                                                # overdispersion
                'rho_h_mu', 'rho_h_sigma', 'rho_h',                     # rho_h
                'f_R_a', 'f_R_b', 'f_R',                                # f_R
                'f_I_mu', 'f_I_sigma', 'f_I',                           # f_I
                'delta_beta_mu',                                        # delta_beta_mu
                'sigma2_0', 'z_0', 'eps_0',                             # AR-GARCH initial condition
                'sum_psi', 'decay_psi',                                 # AR parameters
                'omega', 'a_garch', 'b_garch',                          # GARCH parameters
                ]

# Save original traces
os.makedirs('trace', exist_ok=True)
for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(f'trace/trace-{var}.pdf')
    plt.close()

# Build pair plots
arviz.plot_pair(trace, var_names=["a_garch", "b_garch", "sum_psi", "decay_psi"], divergences=True)
plt.savefig('trace/pairplot-ARGARCH.png', dpi=300)
plt.close()


# Make posterior predictive
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# Predict
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace)

# Save traces and posterior predictive
arviz.to_netcdf(trace, "trace/trace.nc")
arviz.to_netcdf(posterior_predictive, "trace/posterior_predictive.nc")

# Visualise across-season modifier trend + within-season median
fig,ax=plt.subplots(figsize=(8.3, 11.7/5))
# average trend
ax.plot(range(n_modifiers), trace.posterior['delta_beta_mu'].median(dim=['chain', 'draw']).values, color='green')
ax.fill_between(range(n_modifiers),
                trace.posterior['delta_beta_mu'].quantile(dim=['chain', 'draw'], q=0.025).values,
                trace.posterior['delta_beta_mu'].quantile(dim=['chain', 'draw'], q=0.975).values,
                color='green', alpha=0.15)
# individual seasons
for i in range(n_seasons):
    ax.plot(range(n_modifiers), trace.posterior['delta_beta'].median(dim=['chain', 'draw']).values[i,:], color='black', alpha=0.3, linewidth=0.5)
ax.axhline(y=0, color='red', linewidth=0.5)
plt.savefig(f'trace/modifiers.pdf')
plt.close()

# Visualise delta_beta, z, sigma2 and eps per season
for i, season in enumerate(seasons):
    fig,ax=plt.subplots(nrows=4, figsize=(8.3, 11.7))
    # across-season delta_beta trend
    ax[0].plot(range(n_modifiers), trace.posterior['delta_beta_mu'].median(dim=['chain', 'draw']).values, color='green')
    ax[0].fill_between(range(n_modifiers),
                    trace.posterior['delta_beta_mu'].quantile(dim=['chain', 'draw'], q=0.025).values,
                    trace.posterior['delta_beta_mu'].quantile(dim=['chain', 'draw'], q=0.975).values,
                    color='green', alpha=0.15)
    # within-season delta_beta, z, sigma2, eps
    for j, par in enumerate(['delta_beta', 'z', 'sigma2', 'eps']):
        ax[j].plot(range(n_modifiers), trace.posterior[par].median(dim=['chain', 'draw']).values[i,:], color='black', linewidth=0.5)
        ax[j].fill_between(range(n_modifiers),
                trace.posterior[par].quantile(dim=['chain', 'draw'], q=0.025).values[i,:],
                trace.posterior[par].quantile(dim=['chain', 'draw'], q=0.975).values[i,:],
                color='black', alpha=0.15)
        ax[j].set_ylabel(par)
    ax[0].set_title(season)
    plt.savefig(f'trace/{season}_ARMA-GARCH_pars.pdf')
    plt.close()

# Visualise goodnes-of-fit
fig,ax=plt.subplots(nrows=n_seasons, sharex=True, figsize=(8.3, 11.7/5*n_seasons))
for i in range(n_seasons):
    ax[i].plot(ts[i, :], posterior_predictive.posterior_predictive['data'].median(dim=['chain', 'draw']).values[i,:], linewidth=1, color='green')
    ax[i].fill_between(ts[i, :],
                    posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.025).values[i,:],
                    posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.975).values[i,:],
                    color='green', alpha=0.1)
    ax[i].fill_between(ts[i, :],
                    posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.25).values[i,:],
                    posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.75).values[i,:],
                    color='green', alpha=0.2)
    ax[i].scatter(ts[i, :], posterior_predictive.observed_data['data'].values[i,:], marker='o', color='black')
    ax[i].set_title(seasons[i])
plt.savefig(f'trace/plot-fit.pdf')
plt.close()