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

gamma = 1/3.5

# Get US demographics
# ~~~~~~~~~~~~~~~~~~~

state_fips_index = pd.read_csv(os.path.join(abs_dir, 'demography.csv'))[['abbreviation_state', 'name_state', 'fips_state']]
demo = pd.read_csv(os.path.join(abs_dir, 'demography.csv'))['population'].values
n_states = len(state_fips_index)

# Get US incidences
# ~~~~~~~~~~~~~~~~~

# convert to a list of start and enddates (datetime)
n_modifiers = 26
modifier_length = 7
population = 11E6
seasons = ['2023-2024', '2024-2025', '2025-2026']        # script works with only one season
n_seasons = len(seasons)
n_observations = 22
start_calibration_month = 10    # (year X)
end_calibration_month = 3       # (year X+1)
start_calibrations = [datetime(int(season[0:4]), start_calibration_month, 1) for season in seasons]
modifier_reference_dates = [datetime(int(season[0:4]), 10, 15) for season in seasons]
start_simulation = -15 # (October 1)

def get_data(start_calibrations, modifier_reference_dates, n_observations):
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
        df = pd.read_parquet(os.path.join(abs_dir, 'NHSN-HRD_reference-date-2026-03-07_gathered-2026-03-04-17-16-11.parquet.gzip'))
        # convert date column to datetime and fips_state to int
        df['date'] = pd.to_datetime(df['date'], format='ISO8601')
        df['fips_state'] = df['fips_state'].astype(int)
        # slice out variables of interest
        df = df[['date', 'fips_state', 'influenza admissions']]
        # trim temporally
        df = df[((df['date'] > start_calibration) & (df['date'] <= start_calibration+timedelta(weeks=n_observations)))]
        # Backward fill per state (Happens first week of season 2024-2025 in 3 states)
        df['influenza admissions'] = df.groupby('fips_state')['influenza admissions'].bfill()
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
data, dt, ts = get_data(start_calibrations, modifier_reference_dates, n_observations) # (n_season, n_variables, n_observations)


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


# Register with pyMC
# ~~~~~~~~~~~~~~~~~~

# static arguments
args_static = (start_simulation, max(ts[:,-1]), modifier_length)

# Compile forward simulation model
sol_op = SolOp(args_static)
vjp_sol_op = VJPSolOp(args_static)


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

# construct initial differentiable arguments vector
## gradient safe transforms
single_args_diff = jnp.concatenate([
    jnp.array([jnp.log(jnp.exp(beta)-1),           # beta
               jnp.log(jnp.exp(rho_h)-1),          # rho_h
               jnp.log(jnp.exp(f_I)-1),            # f_I
               jnp.log(f_R / (1 - f_R))]),         # f_R
    jnp.arctanh(delta_beta_vals / 0.25)            # delta_beta
])   # shape: (4 + n_modifiers,)
## broadcast across seasons and states
args_diff = jnp.broadcast_to(single_args_diff, (n_seasons, n_states, single_args_diff.shape[0])) # shape: (n_seasons, n_states, n_params)


# stack args_nodiff so two leading axes are seasons, states and the third axes gives the arguments for the season-state combination
gamma_vec = jnp.full((n_seasons, n_states, 1), gamma)
pop_mat = jnp.broadcast_to(jnp.asarray(demo)[None, :, None], (n_seasons, n_states, 1))
ts_mat = jnp.broadcast_to(ts[:, None, :], (n_seasons, n_states, ts.shape[1]))
args_nodiff = np.array(jnp.concatenate([gamma_vec, pop_mat, ts_mat], axis=2))     # shape: (n_seasons, n_states, )  --> convert to numpy otherwise error in pt.as_tensor_variable(args_nodiff) in make_node of pyMC model

# define SSE likelihood
def neg_log_likelihood(args_diff):
    # 1. convert back to untransformed values
    block_1 = jax.nn.softplus(args_diff[:, :, 0:3])        # beta, rho_h, f_I
    block_2 = jax.nn.sigmoid(args_diff[:, :, 3:4])         # f_R
    block_3 = 0.25 * jnp.tanh(args_diff[:, :, 4:])         # delta_beta
    # 2. pack blocks into args_diff
    args_diff = jnp.concatenate([block_1, block_2, block_3], axis=2)
    # 3. run simulation
    pred = jitted_sol_op_multi(args_diff, args_nodiff, args_static)
    # 4. compute SSE loss
    return jnp.sum((data - pred)**2)

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
block_1 = jax.nn.softplus(args_diff[:, :, 0:3])         # beta, rho_h, f_I
block_2 = jax.nn.sigmoid(args_diff[:, :, 3:4])          # f_R
block_3 = 0.25 * jnp.tanh(args_diff[:, :, 4:])          # delta_beta
args_diff = jnp.concatenate([block_1, block_2, block_3], axis=2)  # also back to numpy otherwise initial point will fail

# run simulation
out = jitted_sol_op_multi(args_diff, args_nodiff, args_static)

# inspect result
for s in range(n_states):
    fig, ax = plt.subplots(nrows=1, figsize=(8.7, 11.3/4))
    for i in range(n_seasons):
        ax.plot(dt[i, :], 7*out[i, s, :], color='red', label='pred')
        ax.scatter(dt[i, :], 7*data[i, s, :], marker='o', color='black', label='obs')
    fig.suptitle(f'{state_fips_index.iloc[s]['abbreviation_state']}')
    fig.tight_layout()
    os.makedirs('output/initial-optim', exist_ok=True)
    plt.savefig(f'output/initial-optim/state_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf')
    plt.close(fig)

# store 1D vector per variable so we can start the chains easily
beta_opt = np.array(args_diff[:,:,0])
rho_h_opt = np.array(args_diff[:,:,1])
f_I_opt = np.array(args_diff[:,:,2])
f_R_opt = np.array(args_diff[:,:,3])
delta_beta_opt = np.array(args_diff[:,:,4:])
delta_beta_mu_opt = np.transpose(np.mean(delta_beta_opt, axis=0))


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
    max_per_season_state = data.max(axis=2)
    inv_max = 1.0 / max_per_season_state
    # normalize to mean 1
    normalized = inv_max / inv_max.mean()
    # expand dims for broadcasting across observations
    return normalized[:, :, None]

weights = compute_season_weights(data)

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
    # align alpha with dimensions of value and mu
    alpha = pt.shape_padleft(alpha, value.ndim - 2)
    alpha = pt.shape_padright(alpha, 1)
    # compute log likelihood
    return pt.sum(weights * pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=alpha), value))

def weighted_nb_random(*args, rng=None, size=None):
    """
    Random draws from Negative Binomial for posterior predictive.
    weights are ignored during random draws
    """
    # mu, alpha: tensors -> convert to numpy
    mu_ = np.array(args[0])
    alpha_ = 1/np.array(args[1])
    
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

# Build pyMC probablistic model
with pm.Model() as model:

    # Hyperparameters
    rho_h_mu = pm.Uniform('rho_h_mu', lower=1e-4, upper=1e-2)
    rho_h_sigma = pm.HalfNormal('rho_h_sigma', sigma=1/3)
    f_I_mu = pm.Uniform('f_I_mu', lower=1e-7, upper=1e-3)
    f_I_sigma = pm.HalfNormal('f_I_sigma', sigma=1/3)
    f_R_mu = pm.Beta("f_R_mu", alpha=10, beta=15)
    f_R_kappa_inv = pm.HalfNormal("f_R_kappa_inv", sigma=0.1)
    f_R_a = pm.Deterministic("f_R_a", f_R_mu * (1/f_R_kappa_inv))
    f_R_b  = pm.Deterministic("f_R_b", (1 - f_R_mu) * (1/f_R_kappa_inv))

    # Differentiable parameters
    beta = pt.as_tensor_variable(0.455*np.ones(shape=(n_seasons, n_states)))
    rho_h = pm.LogNormal("rho_h", mu=pt.log(rho_h_mu), sigma=rho_h_sigma, size=(n_seasons, n_states))
    f_I = pm.LogNormal("f_I", mu=pt.log(f_I_mu), sigma=f_I_sigma, size=(n_seasons, n_states))
    f_R = pm.Beta("f_R", alpha=f_R_a, beta=f_R_b, size=(n_seasons, n_states))

    # ------- AR-GARCH modifiers -----------

    # Hyperparameter for delta_beta_temporal
    delta_beta_mu = pm.Normal("delta_beta_mu", mu=0, sigma=0.1, shape=(n_modifiers, n_states))

    # --- AR(1) kernel ---
    # Initial position
    z_0 = pt.zeros([n_seasons, n_states])
    eps_0 = pt.zeros([n_seasons, n_states])
    # Total AR strength (controls overall magnitude)
    psi = pm.Beta("psi", alpha=5, beta=1)
    # sample iid standard normals as shocks
    eta = pm.Normal("eta", mu=0.0, sigma=1.0, shape=(n_modifiers, n_seasons, n_states))
    
    # --- GARCH(1,1) parameters ---                                                                             TO DISABLE GARCH:
    omega = pm.HalfNormal("omega", sigma=0.01/3)
    kappa = pm.Beta("kappa", 3, 1)                                                              
    phi = pm.Beta("phi", 3, 1)                                                                  
    a_garch = pm.Deterministic("a_garch", kappa * phi)                                                          # (a_garch = pt.constant(0.0))
    b_garch = pm.Deterministic("b_garch", kappa * (1 - phi))                                                    # (b_garch = pt.constant(0.0))
    sigma2_0_sigma = pm.HalfNormal('sigma2_0_sigma', sigma=1/3)
    sigma2_0 = pm.LogNormal("sigma2_0", mu=pt.log(omega), sigma=sigma2_0_sigma, shape=(n_seasons, n_states))    # (sigma2_0 = omega * pt.ones(n_seasons))

    # Run AR-GARCH scan over T steps
    (z_seq, sigma2_seq, eps_seq), _ = pytensor.scan(
        fn=step,
        sequences=[eta,],
        outputs_info=[z_0, sigma2_0, eps_0],
        non_sequences=[psi, omega, a_garch, b_garch],
    )

    # Register deterministic variables to inspect later
    delta_beta = pm.Deterministic("delta_beta", z_seq + delta_beta_mu[:, None, :])
    z = pm.Deterministic("z", z_seq)
    sigma2 = pm.Deterministic("sigma2", sigma2_seq)
    eps = pm.Deterministic("eps", eps_seq)

    # concatenate along the last axis
    args_diff = pt.concatenate(
        [beta[:, :, None], rho_h[:, :, None], f_I[:, :, None], f_R[:, :, None], pt.transpose(delta_beta, (1, 2, 0))],
        axis=2
    ) # shape: (n_seasons, n_states, 4 + n_modifiers)

    # Run forward simulation model
    ys = 7*sol_op(args_diff, args_nodiff)
    ys = pt.math.softplus(ys)

    # Compute likelihood
    alpha_inv = pm.HalfNormal("alpha_inv", sigma=0.001, shape=n_states)
    pm.CustomDist("data", ys, 1/alpha_inv, weights, logp=weighted_nb_logp, random=weighted_nb_random, observed=7*data)

# Sample pyMC model
# ~~~~~~~~~~~~~~~~~

with model:
    trace = pm.sample(3, tune=3, chains=1, init='adapt_diag', cores=1, progressbar=True, nuts={'target_accept': 0.8, 'max_treedepth': 8},
                     initvals=1*[{'alpha_inv': 0.01 * pt.ones(n_states), 'delta_beta_mu': delta_beta_mu_opt, 'rho_h': rho_h_opt, 'f_I': f_I_opt, 'f_R': f_R_opt},])

# Generate traces
variables2plot = [
                'alpha_inv',                                            # overdispersion
                'rho_h_mu', 'rho_h_sigma', 'rho_h',                     # rho_h
                'f_R_mu', 'f_R_kappa_inv', 'f_R_a', 'f_R_b', 'f_R',     # f_R
                'f_I_mu', 'f_I_sigma', 'f_I',                           # f_I
                'delta_beta_mu',                                        # delta_beta_mu
                'psi', 'omega', 'kappa', 'phi',                         # AR-GARCH parameters
                'a_garch', 'b_garch', 'sigma2_0', 'sigma2_0_sigma',
                ]

# Save original traces
os.makedirs('output/traces', exist_ok=True)
for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(f'output/traces/trace-{var}.pdf')
    plt.close()

# Build pair plots
arviz.plot_pair(trace, var_names=["kappa", "phi", "omega", "psi"], divergences=True)
plt.savefig('output/traces/pairplot-ARGARCH.png', dpi=300)
plt.close()


# Make posterior predictive
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# Predict
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace)

# Save traces and posterior predictive
arviz.to_netcdf(trace, "output/trace.nc")
arviz.to_netcdf(posterior_predictive, "output/posterior_predictive.nc")

# Visualise across-season modifier trend + within-season median per state
os.makedirs('output/modifiers', exist_ok=True)
# make dates
x = pd.date_range(start=datetime(2000,10,15), periods=n_modifiers, freq='W')
for s in range(n_states):
    fig,ax=plt.subplots(figsize=(8.3, 11.7/5))
    # average trend
    ax.plot(x, 1+trace.posterior['delta_beta_mu'].median(dim=['chain', 'draw']).values[:,s], color='green')
    ax.fill_between(x,
                    1+trace.posterior['delta_beta_mu'].quantile(dim=['chain', 'draw'], q=0.025).values[:,s],
                    1+trace.posterior['delta_beta_mu'].quantile(dim=['chain', 'draw'], q=0.975).values[:,s],
                    color='green', alpha=0.15)
    # individual seasons
    for i in range(n_seasons):
        ax.plot(x, 1+trace.posterior['delta_beta'].median(dim=['chain', 'draw']).values[:,i,s], color='black', alpha=0.3, linewidth=0.5)
    ax.axhline(y=1, color='red', linewidth=0.5)
    # decorations
    fig.suptitle(f'{state_fips_index.iloc[s]['abbreviation_state']}')
    ax.set_ylabel(r'$\Delta \beta_t$')
    ax.set_ylim([0.7, 1.3])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.savefig(f'output/modifiers/modifiers_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf')
    plt.close()

import sys
sys.exit()

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
    plt.savefig(f'output/AR-GARCH_pars/{season}_AR-GARCH_pars.pdf')
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