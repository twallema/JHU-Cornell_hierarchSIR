# standard python libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# all paths defined relative to this file
abs_dir = os.path.dirname(__file__)

# global parameters go here
gamma = 1/3.5
training_name = 'exclude_None'
training_folder = os.path.join(abs_dir, 'output/training')
output_folder = os.path.join(abs_dir, 'output/forecasting')
regions = ['New England', 'Middle Atlantic']

# Get US demographics
# ~~~~~~~~~~~~~~~~~~~

def get_demography(regions=None):
    """
    input
    -----

    regions: list
        A list containing the regions 
    
    output
    ------

    state_fips_index: pd.DataFrame
        contains the abbreviation, name and fips code of states
    
    demography: np.ndarray
        contains the corresponding population
    """
    # get data
    demo = pd.read_csv(os.path.join(abs_dir, 'demography.csv'))
    # slice out right regions
    if regions:
        demo = demo[demo['region_name'].isin(regions)]
    # return index and demography
    return demo[['abbreviation_state', 'name_state', 'fips_state']], demo['population'].values

state_fips_index, demo = get_demography(regions)
n_states = len(demo)

# Get state adjacency matrix
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# get and slice
adj = pd.read_csv(os.path.join(abs_dir, 'adjacency_matrix.csv'), index_col=0)
adj = adj.loc[state_fips_index['abbreviation_state'].values, state_fips_index['abbreviation_state'].values]

# Get US incidence data
# ~~~~~~~~~~~~~~~~~~~~~

# convert to a list of start and enddates (datetime)
seasons = ['2025-2026',]        # script works with only one season
n_seasons = len(seasons)
n_observations = 52             # use all data available in the forecast season
forecast_horizon = 4
# the following variables must match the training script
n_modifiers = 26
modifier_length = 7
start_calibration_month = 10    # (year X)
start_calibrations = [datetime(int(season[0:4]), start_calibration_month, 1) for season in seasons]
modifier_reference_dates = [datetime(int(season[0:4]), 10, 15) for season in seasons]
start_simulation = -15 # (October 1)

def get_data(start_calibrations, modifier_reference_dates, n_observations, forecast_horizon, state_fips=None):
    """
    A function formatting the model's input data

    output:
    -------
    data --> (n_season, n_states, n_observations): number of lab-confirmed influenza admissions
    dates --> (n_season, n_observations): corresponding date
    timesteps --> (n_season, n_observations): data's time index relative to forward simulation model's t=0
    """
    
    data = []
    dates = []
    timesteps = []
    # loop over seasons
    for i, (start_calibration, modifier_reference_date) in enumerate(zip(start_calibrations, modifier_reference_dates)):
        # get the data
        df = pd.read_parquet(os.path.join(abs_dir, 'NHSN-HRD_reference-date-2026-03-14_gathered-2026-03-11-18-20-58.parquet.gzip'))
        # convert date column to datetime and fips_state to int
        df['date'] = pd.to_datetime(df['date'], format='ISO8601')
        df['fips_state'] = df['fips_state'].astype(int)
        # slice out states of interest
        df = df[df['fips_state'].isin(state_fips)]
        # slice out variables of interest
        df = df[['date', 'fips_state', 'influenza admissions']]
        # trim the bottom temporally
        df = df[df['date'] > start_calibration]
        # Backward fill per state (Happens first week of season 2024-2025 in 3 states)
        df['influenza admissions'] = df.groupby('fips_state')['influenza admissions'].bfill()
        # determine the data's end date
        last_existing_date = df['date'].max()
        user_end_date = start_calibration+timedelta(weeks=n_observations)
        if user_end_date <= last_existing_date:
            target_end_date = user_end_date + pd.Timedelta(weeks=forecast_horizon)
        else:
            target_end_date = last_existing_date + pd.Timedelta(weeks=forecast_horizon)
        # generate dataframe encompassing calibration + forecast ranges
        all_dates = pd.date_range(start=df['date'].min(), end=target_end_date, freq='7D')
        all_fips = df['fips_state'].unique()
        full_index = pd.MultiIndex.from_product([all_dates, all_fips], names=['date','fips_state'])
        df = df.set_index(['date', 'fips_state']).reindex(full_index).reset_index()
        # save the data's time index relative to the forward simulation model's t=0 (per season) + dates
        dates.append(df['date'].unique())
        timesteps.append([(d - modifier_reference_date)/timedelta(days=1) for d in df['date'].unique()])
        # extract the data as a (n_states x n_observations) numpy array
        data.append(df.pivot(index="fips_state", columns="date", values="influenza admissions").sort_index().sort_index(axis=1).to_numpy())
    # stack data to (n_season, n_states, n_observations)
    data = np.stack(data, axis=0)
    dates = np.stack(dates, axis=0)
    timesteps = np.stack(timesteps, axis=0)
    
    return data/7, dates, timesteps

# get the data
data, dt, ts = get_data(start_calibrations, modifier_reference_dates, n_observations, forecast_horizon, state_fips_index['fips_state'].values) # (n_season, n_variables, n_observations)

# compute the actual number of observations
n_observations = len(dt[0]) - forecast_horizon

# Get the hyperparameters
# ~~~~~~~~~~~~~~~~~~~~~~~

# get
hyperpars = pd.read_csv(os.path.join(training_folder, f'hyperparameters-{training_name}.csv'))

# unpack
## (global) scalar
rho_global_mean         = hyperpars['rho_global_mean'].unique()[0]
rho_season_sd           = hyperpars['rho_season_sd'].unique()[0]
fI_global_mean          = hyperpars['fI_global_mean'].unique()[0]
fI_season_sd            = hyperpars['fI_season_sd'].unique()[0]
fR_global_mean          = hyperpars['fR_global_mean'].unique()[0]
fR_season_sd            = hyperpars['fR_season_sd'].unique()[0]
omega                   = hyperpars['omega'].unique()[0]
psi_spatial_shocks      = hyperpars['psi_spatial_shocks'].unique()[0]
psi_spatial_modifiers   = hyperpars['psi_spatial_modifiers'].unique()[0]
psi_global_mean         = hyperpars['psi_global_mean'].unique()[0]
kappa_global_mean       = hyperpars['kappa_global_mean'].unique()[0]
phi                     = hyperpars['phi'].unique()[0]
sigma2_0_sigma          = hyperpars['sigma2_0_sigma'].unique()[0]
## (state) vectors
alpha_inv               = hyperpars['alpha_inv'].values
rho_state               = hyperpars['rho_state'].values
fI_state                = hyperpars['fI_state'].values
fR_state                = hyperpars['fR_state'].values
psi_state               = hyperpars['psi_state'].values
kappa_state             = hyperpars['kappa_state'].values

## hypermodifiers
modifier_cols = [c for c in hyperpars.columns if c.startswith("delta_beta_state_mean_")]
delta_beta_state_mean = np.transpose(hyperpars[modifier_cols].to_numpy())

# Define a jax-jitted diffrax differential equation model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    beta, delta_beta_daily, gamma, rho = args
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
    dH = rho * S * FOI - H
    return jnp.array([dS, dI, dR, dH])

# build jax model wrapper
def stop_gradients(x):
    return jax.tree.map(jax.lax.stop_gradient, x)

def sol_op_jax(args_diff, args_nodiff, args_static):
    # unpack differentiable parameters
    beta = args_diff[0]
    rho = args_diff[1]
    fI = args_diff[2]
    fR = args_diff[3]
    delta_beta = args_diff[4:]
    # unpack non-differentiable parameters and block their gradients
    args_nodiff = stop_gradients(args_nodiff)
    gamma = args_nodiff[0]
    population = args_nodiff[1]
    ts = args_nodiff[2:]
    # unpack static arguments
    t0, t1_max, modifier_length = args_static
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
        y0=population * jnp.array([1-fI-fR, fI, fR, 0]),
        args = (beta, delta_beta_daily, gamma, rho),
        saveat=diffrax.SaveAt(ts=list(ts)),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-4)
    )
    return sol.ys[:,-1] # return observed state only


# Vectorized multi-season solver ------------------------------

def sol_op_single(args_diff, args_nodiff, args_static):
    """Wrapper for sol_op_jax to allow vmap."""
    return sol_op_jax(args_diff, args_nodiff, args_static)

# jit the inner ODE model
sol_op_single_jit = jax.jit(sol_op_single, static_argnums=2)

# vmap across the states
state_vmapped = jax.vmap(
    sol_op_single,
    in_axes=(0,0,None),
    out_axes=0
)

# vmap the vmapped states
sol_op_multi = jax.vmap(
    state_vmapped,
    in_axes=(0,0,None),
    out_axes=0
)

# jit again
jitted_sol_op_multi = jax.jit(sol_op_multi, static_argnums=2)

# Define jax VJP (gradient computation) function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def single_vjp(ad, g, an, args_static):
    _, pullback = jax.vjp(
        lambda th: sol_op_jax(th, an, args_static),
        ad
    )
    return pullback(g)[0]


def vjp_sol_op_multi(args_diff, gz, args_nodiff, args_static):

    state_vjp = jax.vmap(
        single_vjp,
        in_axes=(0,0,0,None)
    )

    season_vjp = jax.vmap(
        state_vjp,
        in_axes=(0,0,0,None)
    )

    return season_vjp(args_diff, gz, args_nodiff, args_static)

# jit the gradient 
jitted_vjp_sol_op_multi = jax.jit(vjp_sol_op_multi, static_argnums=3)

# Define the Op and VJPOp classes for the ODE problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SolOp(Op):
    def __init__(self, args_static):
        self.args_static = args_static
        self.vjp_sol_op = VJPSolOp(args_static)

    def make_node(self, args_diff, args_nodiff):
        args_diff = pt.as_tensor_variable(args_diff)
        args_nodiff = pt.as_tensor_variable(args_nodiff)
        return Apply(self, [args_diff, args_nodiff], [pt.tensor3()])

    def perform(self, node, inputs, outputs):
        args_diff, args_nodiff = inputs
        ys = jitted_sol_op_multi(args_diff, args_nodiff, self.args_static)
        outputs[0][0] = np.asarray(ys, dtype=np.float64)

    def grad(self, inputs, output_grads):
        args_diff, args_nodiff = inputs
        (gz,) = output_grads

        grad_wrt_args_diff = self.vjp_sol_op(args_diff, gz, args_nodiff)
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
        ], [pt.tensor3()])                      

    def perform(self, node, inputs, outputs):
        args_diff, gz, args_nodiff = inputs
        # Use the new batched VJP
        grad = jitted_vjp_sol_op_multi(args_diff, gz, args_nodiff, self.args_static)
        # Convert to NumPy array for Theano
        outputs[0][0] = np.asarray(grad, dtype=np.float64)

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

# static arguments
args_static = (start_simulation, float(max(ts[:,:n_observations][:,-1])), modifier_length)

# args diff initial guesses (ballpark estimates) --> #TODO: average out hyperparameters 
beta = 0.455
rho = 0.0025
fI = 1e-4
fR = 0.25
delta_beta_vals = jnp.zeros(n_modifiers)

# compute gradient-safe transformations
args_diff = jnp.concatenate([
                jnp.array([jnp.log(jnp.exp(beta) - 1), jnp.log(jnp.exp(rho) - 1), jnp.log(jnp.exp(fI) - 1), jnp.log(fR / (1 - fR))]),
                jnp.arctanh(delta_beta_vals / 0.25)
            ])
args_diff = jnp.expand_dims(args_diff, 0).repeat(n_seasons, axis=0)

# construct initial differentiable arguments vector
## gradient safe transforms
single_args_diff = jnp.concatenate([
    jnp.array([jnp.log(jnp.exp(beta)-1),           # beta
               jnp.log(jnp.exp(rho)-1),          # rho
               jnp.log(jnp.exp(fI)-1),            # fI
               jnp.log(fR / (1 - fR))]),         # fR
    jnp.arctanh(delta_beta_vals / 0.25)            # delta_beta
])   # shape: (4 + n_modifiers,)
## broadcast across seasons and states
args_diff = jnp.broadcast_to(single_args_diff, (n_seasons, n_states, single_args_diff.shape[0])) # shape: (n_seasons, n_states, n_params)


# stack args_nodiff so two leading axes are seasons, states and the third axes gives the arguments for the season-state combination
gamma_vec = jnp.full((n_seasons, n_states, 1), gamma)
pop_mat = jnp.broadcast_to(jnp.asarray(demo)[None, :, None], (n_seasons, n_states, 1))
ts_mat = jnp.broadcast_to(ts[:, None, :n_observations], (n_seasons, n_states, ts[:,:n_observations].shape[1]))
args_nodiff = np.array(jnp.concatenate([gamma_vec, pop_mat, ts_mat], axis=2))     # shape: (n_seasons, n_states, )  --> convert to numpy otherwise error in pt.as_tensor_variable(args_nodiff) in make_node of pyMC model

# define SSE likelihood
def neg_log_likelihood(args_diff):
    # 1. convert back to untransformed values
    block_1 = jax.nn.softplus(args_diff[:, :, 0:3])        # beta, rho, fI
    block_2 = jax.nn.sigmoid(args_diff[:, :, 3:4])         # fR
    block_3 = 0.25 * jnp.tanh(args_diff[:, :, 4:])         # delta_beta
    # 2. pack blocks into args_diff
    args_diff = jnp.concatenate([block_1, block_2, block_3], axis=2)
    # 3. run simulation
    pred = jitted_sol_op_multi(args_diff, args_nodiff, args_static)
    # 4. compute SSE loss
    return jnp.sum((data[:,:,:n_observations] - pred)**2)

# optimize
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(args_diff)
for i in range(300):
    loss, grads = jax.value_and_grad(neg_log_likelihood)(args_diff)
    updates, opt_state = optimizer.update(grads, opt_state)
    args_diff = optax.apply_updates(args_diff, updates)
    if i % 100 == 0:
        print(i+100, float(loss))

# convert back to untransformed values
block_1 = jax.nn.softplus(args_diff[:, :, 0:3])         # beta, rho, fI
block_2 = jax.nn.sigmoid(args_diff[:, :, 3:4])          # fR
block_3 = 0.25 * jnp.tanh(args_diff[:, :, 4:])          # delta_beta
args_diff = jnp.concatenate([block_1, block_2, block_3], axis=2)  # also back to numpy otherwise initial point will fail

# run simulation
out = jitted_sol_op_multi(args_diff, args_nodiff, args_static)

# inspect result
for s in range(n_states):
    fig, ax = plt.subplots(nrows=1, figsize=(8.7, 11.3/4))
    for i in range(n_seasons):
        ax.plot(dt[i, :n_observations], 7*out[i, s, :], color='red', label='pred')
        ax.scatter(dt[i, :n_observations], 7*data[i, s, :n_observations], marker='o', color='black', label='obs')
    fig.suptitle(f'{state_fips_index.iloc[s]['abbreviation_state']}')
    fig.tight_layout()
    os.makedirs(os.path.join(output_folder, 'initial-optim'), exist_ok=True)
    plt.savefig(os.path.join(output_folder,f'initial-optim/state_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
    plt.close(fig)

# store 2D vector per variable so we can start the chains easily: shape: (n_seasons, n_states)
beta_opt = np.array(args_diff[:,:,0])
rho_opt = np.array(args_diff[:,:,1])
fI_opt = np.array(args_diff[:,:,2])
fR_opt = np.array(args_diff[:,:,3])
delta_beta_opt = np.array(args_diff[:,:,4:])
delta_beta_mu_opt = np.transpose(np.mean(delta_beta_opt, axis=0))

# transform into estimates of global, state and season effects
## rho
log_rho_opt = np.log(rho_opt)
log_rho_global_init = np.mean(log_rho_opt) # global mean
rho_state_init = np.mean(log_rho_opt, axis=0) - log_rho_global_init # state effects (average across seasons, zero-mean)
rho_season_init = np.mean(log_rho_opt, axis=1) - log_rho_global_init # season effects (average across states, zero-mean)
reconstructed = log_rho_global_init + rho_state_init[None, :] + rho_season_init[:, None]
print("Mean log-rho:", log_rho_global_init)
print("Mean reconstruction error:", np.abs(reconstructed - log_rho_opt).mean())
print("Max reconstruction error:", np.abs(reconstructed - log_rho_opt).max())
## fI
log_fI_opt = np.log(fI_opt)
log_fI_global_init = np.mean(log_fI_opt) # global mean
fI_state_init = np.mean(log_fI_opt, axis=0) - log_fI_global_init # state effects (average across seasons, zero-mean)
fI_season_init = np.mean(log_fI_opt, axis=1) - log_fI_global_init # season effects (average across states, zero-mean)
reconstructed = log_fI_global_init + fI_state_init[None, :] + fI_season_init[:, None]
print("Mean log-fI:", log_fI_global_init)
print("Mean reconstruction error:", np.abs(reconstructed - log_fI_opt).mean())
print("Max reconstruction error:", np.abs(reconstructed - log_fI_opt).max())
## fR
from scipy.special import logit
logit_fR_opt = logit(fR_opt)
logit_fR_global_init = np.mean(logit_fR_opt) # global mean
fR_state_init = np.mean(logit_fR_opt, axis=0) - logit_fR_global_init # state effects (average across seasons, zero-mean)
fR_season_init = np.mean(logit_fR_opt, axis=1) - logit_fR_global_init # season effects (average across states, zero-mean)
reconstructed = logit_fR_global_init + fR_state_init[None, :] + fR_season_init[:, None]
print("Mean logit-fR:", logit_fR_global_init)
print("Mean reconstruction error:", np.abs(reconstructed - logit_fR_opt).mean())
print("Max reconstruction error:", np.abs(reconstructed - logit_fR_opt).max())

# Register with pyMC
# ~~~~~~~~~~~~~~~~~~

# (make it so that it runs to end of forecast horizon)
 
# static arguments
args_static = (start_simulation, float(max(ts[:,-1])), modifier_length)

# non-differentiable arguments
gamma_vec = jnp.full((n_seasons, n_states, 1), gamma)
pop_mat = jnp.broadcast_to(jnp.asarray(demo)[None, :, None], (n_seasons, n_states, 1))
ts_mat = jnp.broadcast_to(ts[:, None, :], (n_seasons, n_states, ts.shape[1]))
args_nodiff = np.array(jnp.concatenate([gamma_vec, pop_mat, ts_mat], axis=2))     # shape: (n_seasons, n_states, )  --> convert to numpy otherwise error in pt.as_tensor_variable(args_nodiff) in make_node of pyMC model

# Compile forward simulation model as a pyMC probablistic node
sol_op = SolOp(args_static)
vjp_sol_op = VJPSolOp(args_static)


# Build tempored NB distribution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# computed tempered likelihood weights
def compute_season_weights(data):
    """
    Compute weights so each season-state contributes equally.

    Parameters
    ----------
    data : ndarray (n_seasons, n_states, n_observations)

    Returns
    -------
    weights : np.ndarray, shape (n_seasons, n_states, 1)
    """
    # max over observations per season-state
    max_per_season_state = np.sqrt(data.mean(axis=2))
    inv_max = 1.0 / max_per_season_state
    # normalize to mean 1
    normalized = inv_max / inv_max.mean()
    # expand dims for broadcasting across observations
    return normalized[:, :, None]

weights = compute_season_weights(data[:,:,:n_observations])

# tempered negative binomial likelihood
def weighted_nb_logp(value, mu, alpha, weights):
    """
    Weighted Negative Binomial log-probability.

    Parameters
    ----------
    value : observed counts
        shape (n_seasons, n_states, observations)

    mu : predicted mean
        shape (n_seasons, n_states, observations)

    alpha : NB dispersion parameter
        shape (n_states,)

    weights : season weights
        shape (n_seasons, n_states, 1)
    """

    # move state axis to the end so alpha (n_states,) broadcasts correctly
    mu = mu.dimshuffle(0, 2, 1)
    value = value.dimshuffle(0, 2, 1)
    weights = weights.dimshuffle(0, 2, 1)

    return pt.sum(weights * pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=alpha), value))

def weighted_nb_random(*args, rng=None, size=None):
    """
    Random draws from Negative Binomial for posterior predictive.
    weights are ignored during random draws
    """
    # mu, alpha: tensors -> convert to numpy
    mu_ = np.array(args[0])
    alpha_ = 1/np.array(args[1])

    # remove pyMC broadcast axes
    alpha_ = alpha_.reshape(-1)

    # broadcast to mu
    alpha_ = alpha_[None, :, None]

    # size: PyMC passes shape of batch/draws
    return rng.negative_binomial(n=1/alpha_, p=1/(1 + mu_ * alpha_), size=size)


# Build pyMC model
# ~~~~~~~~~~~~~~~~

# AR(1)-GARCH(1,1) step function
def step(eta_t, prev_z, prev_sigma2, prev_eps, psi, omega, a_garch, b_garch):
    # --- GARCH(1,1) short-term shocks innovation scale ---
    sigma2 = omega + a_garch * (prev_eps ** 2) + b_garch * prev_sigma2
    eps = eta_t * pt.sqrt(sigma2)
    # --- AR(1) short-term shocks ---
    z = psi * prev_z + eps
    return z, sigma2, eps

# construct coordinates
coords = {
    "state": state_fips_index['abbreviation_state'].values,
    "season": seasons,
    "modifier": np.arange(n_modifiers)
}

# Build pyMC probablistic model
with pm.Model(coords=coords) as model:

    # Hyperparameters '<parameter>_<level>_<type>' with level: {global, state, season} and type: {mean, sd, offset}

    ## transmission coefficient: beta (fixed)
    beta = pt.as_tensor_variable(0.455*np.ones(shape=(n_seasons,n_states)))

    ## ascertainment: rho
    ### global (rho_global_mean)
    ### state (rho_state)
    ### season (rho_season_sd)
    rho_season_raw = pm.Normal("rho_season_raw", 0, 1, dims="season")
    log_rho = pt.log(rho_global_mean) + pt.log(rho_state)[None, :] + rho_season_sd * rho_season_raw[:, None]
    rho = pm.Deterministic("rho", pt.exp(log_rho))

    ## initial infected: fI
    ### global (fI_global_mean)
    ### state (fI_state)
    ### season (fI_season_sd)
    fI_season_raw = pm.Normal("fI_season_raw", 0, 1, dims="season")
    log_fI = pt.log(fI_global_mean) + pt.log(fI_state)[None, :] + fI_season_sd * fI_season_raw[:, None]
    fI = pm.Deterministic("fI", pt.exp(log_fI))

    ## initial recovered: fR
    ### global (fR_global_mean)
    ### state (fR_state)
    ### season (fR_season_sd)
    fR_season_raw = pm.Normal("fR_season_raw", 0, 1, dims="season")
    logit_fR = pm.math.logit(fR_global_mean) + pt.log(fR_state)[None, :] + fR_season_sd * fR_season_raw[:, None]
    fR = pm.Deterministic("fR", pm.math.sigmoid(logit_fR))

    # ------- AR-GARCH modifiers -----------

    # Spatial correlation ('psi_spatial_shocks' hyperparameter)
    W = pt.as_tensor_variable(adj.values)
    D = pt.diag(pt.sum(W, axis=1))
    Q_shocks = D - psi_spatial_shocks * W + 1e-6 * pt.eye(n_states)
    L_Q_shocks = pt.slinalg.cholesky(Q_shocks)
    L_cov_shocks = pt.slinalg.solve(L_Q_shocks, pt.eye(n_states))

    # Hyperparameter for delta_beta_temporal (delta_beta_state_mean hyperparameter, shape: n_modifiers x n_states)

    # --- AR(1) kernel (season axis removed) ---
    # Initial position
    z_0 = pt.zeros([n_states,])
    eps_0 = pt.zeros([n_states,])
    # Steady state noise ('omega' hyperparameter)
    # Total AR persistence
    ## global (psi_global_mean)
    ## state (psi_state)
    psi = pm.Deterministic("psi", pm.math.sigmoid(pm.math.logit(psi_global_mean) + pt.log(psi_state)))
    # sample iid standard normals as shocks
    eta_raw = pm.Normal("eta_raw", mu=0.0, sigma=1.0, dims=("modifier","state"))
    # correlate them across space using the precision matrix 
    eta = pm.Deterministic("eta", pt.einsum("ij,mj->mi", L_cov_shocks, eta_raw))    # shape: (modifier x state)

    # --- GARCH(1,1) parameters ---                                                                             
    # Total noise persistence
    ## global (kappa_global_mean)
    ## state (kappa_state)
    kappa = pm.Deterministic("kappa", pm.math.sigmoid(pm.math.logit(kappa_global_mean) + pt.log(kappa_state)))      
    # Split between a and b (phi & sigma2_0_sigma hyperparameter)                                                                                                             
    a_garch = pm.Deterministic("a_garch", kappa * phi)                                                          
    b_garch = pm.Deterministic("b_garch", kappa * (1 - phi))                           
    sigma2_0 = pm.LogNormal("sigma2_0", mu=pt.log(omega/(1-kappa)), sigma=sigma2_0_sigma, dims="state")

    # Run AR-GARCH scan over T steps
    (z_seq, sigma2_seq, eps_seq), _ = pytensor.scan(
        fn=step,
        sequences=[eta,],
        outputs_info=[z_0, sigma2_0, eps_0],
        non_sequences=[psi, omega, a_garch, b_garch],
    )

    # Register deterministic variables to inspect later
    delta_beta = pm.Deterministic("delta_beta", delta_beta_state_mean + z_seq)
    z = pm.Deterministic("z", z_seq)
    sigma2 = pm.Deterministic("sigma2", sigma2_seq)
    eps = pm.Deterministic("eps", eps_seq)

    # concatenate parameters along last axis (n_seasons, n_states, n_parameters)
    args_diff = pt.concatenate(
        [beta[:, :, None], rho[:, :, None], fI[:, :, None], fR[:, :, None], pt.transpose(delta_beta, (1, 0))[None, :, :]],
        axis=2
    )

    # Run forward simulation model
    ys = 7*sol_op(args_diff, args_nodiff)
    ys = pt.math.softplus(ys)

    # Compute likelihood (alpha_inv hyperparameter)
    pm.CustomDist("obs", ys[:,:,:n_observations], 1/alpha_inv, weights, logp=weighted_nb_logp, random=weighted_nb_random, observed=7*data[:,:,:n_observations])

# Sample pyMC model
# ~~~~~~~~~~~~~~~~~

n_chains = 6
with model:
    # run sampler without tuning
    trace = pm.sample(100, tune=100, chains=n_chains, init='adapt_diag', cores=1, progressbar=True)

print(f"Step size post-tuning: {trace.sample_stats.step_size_bar.values}")

# manual burn
n_burn = 0
trace = trace.isel(draw=slice(n_burn, None))

# Generate traces
variables2plot = ['rho', 'fI', 'fR', 'psi', 'kappa', 'sigma2_0']

# Save original traces
os.makedirs(os.path.join(output_folder, 'traces'), exist_ok=True)
for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(os.path.join(output_folder, f'traces/trace-{var}.pdf'))
    plt.close()

# Make posterior predictive
# ~~~~~~~~~~~~~~~~~~~~~~~~~

with model:

    # geometric random walk per state
    grw_innov = pm.Normal("grw_innov", mu=0, sigma=0.375, shape=(n_states, forecast_horizon))         # tune by LOOCV on WIS (currently set to NC stationary GRW baseline model optimal)
    ys_future_rw = ys[:, :, n_observations:] * pt.exp(pt.cumsum(grw_innov, axis=1)[None, :, :])

    # forecast 
    pred = pm.NegativeBinomial("pred", mu=ys_future_rw, alpha=1/alpha_inv[None, :, None])

    # sample
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["obs", "pred"])


# Save traces and posterior predictive
arviz.to_netcdf(trace, os.path.join(output_folder, "trace.nc"))
arviz.to_netcdf(posterior_predictive, os.path.join(output_folder, "posterior_predictive.nc"))

# Visualise goodness-of-fit
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# Visualise
dates_obs = dt[0,:n_observations]
dates_pred = dt[0,n_observations:]
for s in range(n_states):
    fig,ax=plt.subplots()
    ## training
    ax.plot(dates_obs, posterior_predictive.posterior_predictive['obs'].median(dim=['chain', 'draw']).values[0,s,:], linewidth=1, color='black')
    ax.fill_between(dates_obs,
                    posterior_predictive.posterior_predictive['obs'].quantile(dim=['chain', 'draw'], q=0.025).values[0,s,:],
                    posterior_predictive.posterior_predictive['obs'].quantile(dim=['chain', 'draw'], q=0.975).values[0,s,:],
                    color='black', alpha=0.1)
    ax.fill_between(dates_obs,
                    posterior_predictive.posterior_predictive['obs'].quantile(dim=['chain', 'draw'], q=0.025).values[0,s,:],
                    posterior_predictive.posterior_predictive['obs'].quantile(dim=['chain', 'draw'], q=0.75).values[0,s,:],
                    color='black', alpha=0.1)    
    ax.scatter(dates_obs, posterior_predictive.observed_data['obs'].values[0,s,:], marker='o', color='black')
    ## forecast
    ax.plot(dates_pred, posterior_predictive.posterior_predictive['pred'].median(dim=['chain', 'draw']).values[0,s,:], linewidth=1, color='red')
    ax.fill_between(dates_pred,
                    posterior_predictive.posterior_predictive['pred'].quantile(dim=['chain', 'draw'], q=0.025).values[0,s,:],
                    posterior_predictive.posterior_predictive['pred'].quantile(dim=['chain', 'draw'], q=0.975).values[0,s,:],
                    color='red', alpha=0.1)
    ax.fill_between(dates_pred,
                    posterior_predictive.posterior_predictive['pred'].quantile(dim=['chain', 'draw'], q=0.25).values[0,s,:],
                    posterior_predictive.posterior_predictive['pred'].quantile(dim=['chain', 'draw'], q=0.75).values[0,s,:],
                    color='red', alpha=0.1)    
    fig.suptitle(f'{state_fips_index.iloc[s]['abbreviation_state']}')
    fig.tight_layout()
    os.makedirs(os.path.join(output_folder, 'goodness-fit'), exist_ok=True)
    plt.savefig(os.path.join(output_folder,f'goodness-fit/state_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
    plt.close(fig)