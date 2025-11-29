import numpy as np
import matplotlib.pyplot as plt

import pytensor
import pytensor.tensor as pt
pytensor.config.cxx = '/usr/bin/clang++'
pytensor.config.on_opt_error = "ignore"
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

import jax
import jax.numpy as jnp

import pymc as pm
import pymc.sampling.jax
import arviz


# Generate a synthetic dataset - exponential growth with overdisperion
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Parameters 
alpha = 0.05
t_d = 10
n_timesteps = 90
# Sample data
ts = np.linspace(start=0, stop=n_timesteps-1, num=n_timesteps)
data = np.random.negative_binomial(1/alpha, (1/alpha)/(np.exp(ts*np.log(2)/t_d) + (1/alpha)))


# Define and solve a diffrax differential equation and wrap it inside jax jit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# define delta_beta[t] modifier function
from jax.scipy.signal import convolve

def smooth_with_gaussian(vec, sigma=2.0):
    window_size = 15
    # Build a Gaussian kernel
    x = jnp.linspace(-3, 3, window_size)
    kern = jnp.exp(-0.5 * (x/sigma)**2)
    kern = kern / kern.sum()
    return convolve(vec, kern, mode="same")


def make_delta_beta_daily(delta_beta, duration, t0, t1, sigma=1):
    """
    Parameters
    ----------
    delta_beta : array (K,)
        Values for each block.
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
import diffrax

def stop_gradients(x):
    return jax.tree.map(jax.lax.stop_gradient, x)

def sol_op_jax(args_diff, args_nodiff, args_static):
    # unpack differentiable parameters
    beta = args_diff[0]
    rho_h = args_diff[1]
    f_I = args_diff[2]
    f_R = args_diff[3]
    delta_beta = args_diff[3:]
    # unpack non-differentiable parameters and block gradients
    args_nodiff = stop_gradients(args_nodiff)
    gamma = args_nodiff[0]
    # unpack static arguments
    t0, t1, ts, modifier_length, population = args_static
    # evaluate modifiers
    delta_beta_daily = make_delta_beta_daily(delta_beta, modifier_length, t0, t1)
    # wrap ODE rhs
    term = diffrax.ODETerm(SIR_vector_field)
    # solve ODE
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=population * jnp.array([1-f_I-f_R, f_I, f_R, 0]),
        args = (beta, delta_beta_daily, gamma, rho_h),
        saveat=diffrax.SaveAt(ts=list(ts)),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-4)
    )
    return sol.ys[:,-1] # return observed state only

jitted_sol_op_jax = jax.jit(sol_op_jax, static_argnums=2)


# Define VJP function
# ~~~~~~~~~~~~~~~~~~~

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


# Build pyMC model
# ~~~~~~~~~~~~~~~~

# Define static forward simulation model parameters
population = 11E6
n_modifiers = 5
modifier_length = 14
args_static = (ts[0], ts[-1], tuple(ts), modifier_length, population)

# Compile forward simulation model
sol_op = SolOp(args_static)
vjp_sol_op = VJPSolOp(args_static)

# Build pyMC probablistic model
with pm.Model() as model:

    # Differentiable parameters (those we wish to calibrate)
    beta = pm.Truncated("beta", pm.LogNormal.dist(mu=-2, sigma=0.25), lower=0, upper=1) # E[X] = 0.14, SD[X] = 0.035
    delta_beta = pm.Truncated("delta_beta", pm.Normal.dist(mu=0, sigma=0.01), size=n_modifiers, lower=-0.05, upper=0.05)
    rho_h = pm.LogNormal("rho_h", mu=-6, sigma=0.25)
    f_I = pm.LogNormal("f_I", mu=-10, sigma=0.5)
    f_R = pm.Beta("f_R", alpha=5, beta=5)

    # Non-Differentiable parameters (those we do not wish to calibrate) 
    gamma = pt.as_tensor_variable([1/10,])
    
    # Build forward simulation arguments
    args_diff = pt.concatenate([beta.reshape((1,)), rho_h.reshape((1,)), f_I.reshape((1,)), f_R.reshape((1,)), delta_beta]) # flatten all inputs
    args_nodiff = gamma

    # Run model
    ys = sol_op(args_diff, args_nodiff)
    ys = pt.math.softplus(ys)

    # Likelihood
    alpha = pm.HalfNormal("alpha", 0.05)
    data = pm.NegativeBinomial("data", mu=ys, alpha=1/alpha, observed=data)


# Sample pyMC model
# ~~~~~~~~~~~~~~~~~

with model:
    trace = pm.sample(100, tune=100, chains=2, init='jitter+adapt_diag', cores=1, progressbar=True, initvals=2*[{'alpha': 0.10, 'beta': 0.25, 'rho_h': 0.004, 'f_I': 1E-4, 'f_R': 0.25},])

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