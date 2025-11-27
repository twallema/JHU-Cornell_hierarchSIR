import pytensor
import pymc as pm
import pytensor.tensor as pt
pytensor.config.cxx = '/usr/bin/clang++'
pytensor.config.on_opt_error = "ignore"


import numpy as np
import matplotlib.pyplot as plt
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

import jax
import jax.numpy as jnp

import pymc.sampling.jax
import arviz


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
saveat = diffrax.SaveAt(ts=[0, 10, 20, 30, 40, 50])

def sol_op_jax(args_diff, args_nodiff):
    # unpack parameters
    beta, gamma = args_diff
    t0, t1 = args_nodiff
    # wrap ODE rhs
    term = diffrax.ODETerm(SIR_vector_field)
    # solve
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Dopri5(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=jnp.array([10e6, 1, 0]),
        args = (beta, gamma),
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-12)
    )
    return sol.ys[:,2] # return R state only

jitted_sol_op_jax = jax.jit(sol_op_jax, static_argnums=1)


# Define VJP function
# ~~~~~~~~~~~~~~~~~~~

def vjp_sol_op_jax(args_diff, gz, args_nodiff):
    _, vjp_fn = jax.vjp(lambda th: sol_op_jax(th, args_nodiff), args_diff)
    return vjp_fn(gz)[0]

jitted_vjp_sol_op_jax = jax.jit(vjp_sol_op_jax, static_argnums=2)


# Define the Op and VJPOp classes for the ODE problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SolOp(Op):
    def __init__(self, args_nodiff):
        self.args_nodiff = args_nodiff  # store non-differentiated constants here

    def make_node(self, args_diff):
        outputs = [pt.vector(dtype="float64")]
        return Apply(self, [pt.as_tensor_variable(args_diff)], outputs)

    def perform(self, node, inputs, outputs):
        (args_diff,) = inputs
        result = jitted_sol_op_jax(args_diff, self.args_nodiff)
        outputs[0][0] = np.asarray(result, dtype="float64")

    def grad(self, inputs, output_grads):
        (args_diff,) = inputs
        (gz,) = output_grads
        return [vjp_sol_op(args_diff, gz)]


class VJPSolOp(Op):
    def __init__(self, args_nodiff):
        self.args_nodiff = args_nodiff

    def make_node(self, args_diff, gz):
        inputs = [pt.as_tensor_variable(args_diff), pt.as_tensor_variable(gz)]
        outputs = [inputs[0].type()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        args_diff, gz = inputs
        result = jitted_vjp_sol_op_jax(args_diff, gz, self.args_nodiff)
        outputs[0][0] = np.asarray(result, dtype="float64")

# Register with jax
# ~~~~~~~~~~~~~~~~~


@jax_funcify.register(SolOp)
def sol_op_jax_funcify(op, **kwargs):
    return lambda args_diff: sol_op_jax(args_diff, op.args_nodiff)

@jax_funcify.register(VJPSolOp)
def vjp_sol_op_jax_funcify(op, **kwargs):
    return lambda args_diff, gz: vjp_sol_op_jax(args_diff, gz, op.args_nodiff)

# Build pyMC model
# ~~~~~~~~~~~~~~~~

time = [0, 10, 20, 30, 40, 50]
data = [0, 12, 45, 320, 1400, 9000]

# Non-Differentiable model parameters
args_nodiff = (time[0], time[-1])

# Compile model
sol_op = SolOp(args_nodiff)
vjp_sol_op = VJPSolOp(args_nodiff)


with pm.Model() as model:
    # Differentiable parameters
    beta = pm.Normal("beta", 0.5, 0.05)
    gamma = pt.as_tensor_variable([1/3,])
    args_diff = pt.concatenate([beta.ravel(), gamma.ravel()])
    # Run model
    ys = sol_op(args_diff)
    ys = pt.math.softplus(ys)
    # Likelihood
    alpha = pm.HalfNormal("alpha", 1)
    data = pm.NegativeBinomial("data", mu=ys, alpha=1/alpha, observed=data)

# Sample pyMC model
# ~~~~~~~~~~~~~~~~~

with model:
    trace = pm.sample(100, tune=100, chains=2, init='jitter+adapt_diag', cores=1, progressbar=True)

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
ax.plot(time, posterior_predictive.posterior_predictive['data'].median(dim=['chain', 'draw']).values, linewidth=1, color='red')
ax.fill_between(time,
                posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.025),
                posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.975),
                color='red', alpha=0.1)
ax.scatter(time, posterior_predictive.observed_data['data'].values, marker='o', color='black')
plt.show()
plt.close()