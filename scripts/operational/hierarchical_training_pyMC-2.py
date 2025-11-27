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

# define ODE rhs
def SIR_vector_field(t, y, args):
    S, I, R = y
    N = S + I + R
    beta, gamma = args
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return jnp.array([dS, dI, dR])


import diffrax

def sol_op_jax(args_diff, args_nodiff, args_static):
    # unpack differentiable parameters
    beta = args_diff[0]
    # unpack non-differentiable parameters and block gradients
    gamma = jax.lax.stop_gradient(args_nodiff[0])
    # unpack static arguments
    t0, t1, ts = args_static
    # wrap ODE rhs
    term = diffrax.ODETerm(SIR_vector_field)
    # solve ODE
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Dopri5(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=jnp.array([10e6, 1, 0]),
        args = (beta, gamma),
        saveat=diffrax.SaveAt(ts=list(ts)),
        stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-12)
    )
    return sol.ys[:,2] # return R state only

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

# Non-Differentiable model parameters
args_static = (ts[0], ts[-1], tuple(ts))

# Compile model
sol_op = SolOp(args_static)
vjp_sol_op = VJPSolOp(args_static)

with pm.Model() as model:
    # Differentiable parameters (those we wish to calibrate)
    beta = pm.Normal("beta", 0.5, 0.05)
    args_diff = pt.stack([beta,])
    # Non-Differentiable parameters (those we do not wish to calibrate) 
    gamma = pt.as_tensor_variable([1/3,])
    args_nodiff = gamma
    # Run model
    ys = sol_op(args_diff, args_nodiff)
    ys = pt.math.softplus(ys)
    # Likelihood
    alpha = pm.HalfNormal("alpha", 0.01)
    data = pm.NegativeBinomial("data", mu=ys, alpha=1/alpha, observed=data)

# Sample pyMC model
# ~~~~~~~~~~~~~~~~~

with model:
    trace = pm.sample(1000, tune=1000, chains=2, init='jitter+adapt_diag', cores=1, progressbar=True)

# Generate traces
arviz.plot_trace(trace, var_names=['alpha', 'beta']) 
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