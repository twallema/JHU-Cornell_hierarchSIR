# standard python libraries
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
n_modifiers = 12
modifier_length = 15
population = 11E6
seasons = ['2023-2024', '2024-2025']        # script works with only one season
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
    S = jnp.clip(S, 0.0, None)
    I = jnp.clip(I, 0.0, None)
    R = jnp.clip(R, 0.0, None)
    H = jnp.clip(H, 0.0, None)
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

# stack per season args_diff
args_diff = jnp.stack([
                jnp.concatenate([jnp.array([0.45, 0.0025, 1E-4, 0.3]), jnp.zeros(n_modifiers)]),
                jnp.concatenate([jnp.array([0.48, 0.0025, 1E-4, 0.3]), jnp.zeros(n_modifiers)])
            ])
# stack per season argsnodiff
args_nodiff = jnp.stack([
                jnp.concatenate([jnp.array([1/3.5,]), jnp.array(ts[0,:])]),
                jnp.concatenate([jnp.array([1/3.5,]), jnp.array(ts[1,:])]),
            ])
# static arguments
args_static = (start_simulation, max(ts[:,-1]), modifier_length, population)

# simulate model
out = jitted_sol_op_multi(args_diff, args_nodiff, args_static)

# visualise
fig,ax=plt.subplots(nrows=2)
ax[0].plot(ts[0,:], out[0,:], color='red')
ax[1].plot(ts[1,:], out[1,:], color='red')
plt.show()
plt.close()

print(out)

import sys
sys.exit()

# Define jax VJP (gradient computation) function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def vjp_sol_op_jax(args_diff, gz, args_nodiff, args_static):
    _, vjp_fn = jax.vjp(lambda th: sol_op_jax(th, args_nodiff, args_static), args_diff)
    return vjp_fn(gz)[0]

jitted_vjp_sol_op_jax = jax.jit(vjp_sol_op_jax, static_argnums=3)


# Define the Op and VJPOp classes for the ODE problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SolOp(Op):
    def __init__(self, args_static):
        self.args_static = args_static

    def make_node(self, args_diff, args_nodiff):
        args_diff = pt.as_tensor_variable(args_diff)
        args_nodiff = pt.as_tensor_variable(args_nodiff)
        return Apply(self, [args_diff, args_nodiff], [pt.vector()])

    def perform(self, node, inputs, outputs):
        args_diff, args_nodiff = inputs
        ys = jitted_sol_op_jax(args_diff, args_nodiff, self.args_static)
        outputs[0][0] = np.asarray(ys)

    def grad(self, inputs, output_grads):
        """
        Return symbolic gradients for the inputs of this Op.
        The return list must have the same length as `inputs`.
        """
        args_diff, args_nodiff = inputs
        (gz,) = output_grads

        # vjp_sol_op is the VJPSolOp instance you created at module scope
        # It builds an Apply node that computes the gradient w.r.t. args_diff.
        grad_wrt_args_diff = vjp_sol_op(args_diff, gz, args_nodiff)

        # We block gradients through args_nodiff: return a zero tensor of the same shape.
        grad_wrt_args_nodiff = pt.zeros_like(args_nodiff)

        return [grad_wrt_args_diff, grad_wrt_args_nodiff]

class VJPSolOp(Op):
    def __init__(self, args_static):
        self.args_static = args_static

    def make_node(self, args_diff, gz, args_nodiff):
        return Apply(self, [
            pt.as_tensor_variable(args_diff),
            pt.as_tensor_variable(gz),
            pt.as_tensor_variable(args_nodiff)
        ], [pt.vector()])

    def perform(self, node, inputs, outputs):
        args_diff, gz, args_nodiff = inputs
        grad = jitted_vjp_sol_op_jax(args_diff, gz, args_nodiff, self.args_static)
        outputs[0][0] = np.asarray(grad)


# Register with jax
# ~~~~~~~~~~~~~~~~~

@jax_funcify.register(SolOp)
def sol_op_jax_funcify(op, **kwargs):
    return lambda args_diff, args_nodiff: sol_op_jax(args_diff, args_nodiff, op.args_static)

@jax_funcify.register(VJPSolOp)
def vjp_sol_op_jax_funcify(op, **kwargs):
    return lambda args_diff, gz, args_nodiff: vjp_sol_op_jax(args_diff, gz, args_nodiff, op.args_static)


# Pre-optimize the forward simulation model's parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# args diff (ballpark estimates)
beta = 0.46
rho_h = 0.0025
f_I = 1e-4
f_R = 0.4
delta_beta_vals = jnp.zeros(n_modifiers)

# args nodiff
args_nodiff = jnp.array([1/3.5])    # gamma

# args static
t0 = start_simulation
t1 = ts[-1]
args_static = (t0, t1, tuple(ts), modifier_length, population)

# define SSE likelihood
def neg_log_likelihood(theta_raw):
    # 1. Transform raw params -> constrained params
    beta     = jax.nn.softplus(theta_raw[0])        # > 0
    rho_h    = jax.nn.softplus(theta_raw[1])        # > 0
    f_I      = jax.nn.softplus(theta_raw[2])        # > 0
    f_R      = jax.nn.sigmoid(theta_raw[3])         # (0, 1)
    delta_beta = 0.25 * jnp.tanh(theta_raw[4:])     # bounded (-0.25, 0.25)
    # 2. Pack args_diff
    args_diff_new = jnp.concatenate([
        jnp.array([beta, rho_h, f_I, f_R]),
        delta_beta
    ])
    # 3. Run simulation
    pred = jitted_sol_op_jax(args_diff_new, args_nodiff, args_static)
    # 4. Compute SSE loss
    return jnp.sum((data.squeeze() - pred)**2)


# optimize
params = jnp.concatenate([
    jnp.array([jnp.log(jnp.exp(beta) - 1), jnp.log(jnp.exp(rho_h) - 1), jnp.log(jnp.exp(f_I) - 1), jnp.log(f_R / (1 - f_R))]),
    jnp.arctanh(delta_beta_vals / 0.25)
])
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(params)
for i in range(1000):
    loss, grads = jax.value_and_grad(neg_log_likelihood)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 100 == 0:
        print(i, float(loss))

# assign to variables
beta_opt     = jax.nn.softplus(params[0]).item()
rho_h_opt    = jax.nn.softplus(params[1]).item()
f_I_opt      = jax.nn.softplus(params[2]).item()
f_R_opt      = jax.nn.sigmoid(params[3]).item()
delta_beta_opt    = np.array(0.25 * jnp.tanh(params[4:]))
params = jnp.concatenate([jnp.array([beta_opt, rho_h_opt, f_I_opt, f_R_opt]), delta_beta_opt])

# run simulation
out = jitted_sol_op_jax(params, args_nodiff, args_static)

# inspect result
fig,ax=plt.subplots()
ax.plot(ts, 7*out, color='red')
ax.scatter(ts, 7*data.squeeze(), marker='o', color='black')
ax.set_title("Pre-sampling goodness-of-fit")
plt.show()
plt.close()


# Build pyMC model
# ~~~~~~~~~~~~~~~~

# Define static forward simulation model parameters
args_static = (start_simulation, ts[-1], tuple(ts), modifier_length, population)

# Compile forward simulation model
sol_op = SolOp(args_static)
vjp_sol_op = VJPSolOp(args_static)

# Build pyMC probablistic model
with pm.Model() as model:

    # Differentiable parameters (those we wish to calibrate)
    beta = pm.Truncated("beta", pm.LogNormal.dist(mu=-0.75, sigma=0.05), lower=0, upper=1) # E[X] = 0.46, SD[X] = 0.04
    delta_beta = pm.Truncated("delta_beta", pm.Normal.dist(mu=0, sigma=0.1), size=n_modifiers, lower=-0.5, upper=0.5)
    rho_h = pm.LogNormal("rho_h", mu=-6, sigma=0.5)
    f_I = pm.LogNormal("f_I", mu=-10, sigma=1)
    f_R = pm.Beta("f_R", alpha=5, beta=5)

    # Non-Differentiable parameters (those we do not wish to calibrate) 
    gamma = pt.as_tensor_variable([1/3.5,])
    
    # Build forward simulation arguments
    args_diff = pt.concatenate([beta.reshape((1,)), rho_h.reshape((1,)), f_I.reshape((1,)), f_R.reshape((1,)), delta_beta]) # flatten all inputs
    args_nodiff = gamma

    # Run model
    ys = 7*sol_op(args_diff, args_nodiff)
    ys = pt.math.softplus(ys)

    # Likelihood
    alpha = pm.HalfNormal("alpha", sigma=0.005)
    data = pm.NegativeBinomial("data", mu=ys, alpha=1/alpha, observed=7*data)


# Sample pyMC model
# ~~~~~~~~~~~~~~~~~

with model:
    trace = pm.sample(100, tune=100, chains=2, init='adapt_diag', cores=1, progressbar=True, initvals=2*[{'alpha': 0.001, 'beta': beta_opt, 'delta_beta': delta_beta_opt, 'rho_h': rho_h_opt, 'f_I': f_I_opt, 'f_R': f_R_opt},])

# Generate traces
arviz.plot_trace(trace, var_names=['alpha', 'beta', 'delta_beta', 'rho_h', 'f_I', 'f_R']) 
plt.show()
plt.close()


# Make posterior predictive
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# Predict
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace)


# Visualise
fig,ax=plt.subplots()
ax.plot(ts, posterior_predictive.posterior_predictive['data'].median(dim=['chain', 'draw']).values, linewidth=1, color='red')
ax.fill_between(ts,
                posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.025),
                posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.975),
                color='red', alpha=0.1)
ax.scatter(ts, posterior_predictive.observed_data['data'].values, marker='o', color='black')
plt.show()
plt.close()