"""
Bayesian version of the learning model
"""

# packages needed
import os
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz
import matplotlib.pyplot as plt


# get data
data = pd.read_csv('accuracy.csv', parse_dates=True)
df = data.copy()

# format data
df["training_horizon"] = (
    df["hyperparameters"]
    .astype(str)
    .str[-1]
    .astype(int)
)
df["log_rel_wis"] = np.log(df["relative_WIS_nodrift"])
df["ED"] = (df["ED_visits"] == True).astype(int)
df["model"] = df["model"].astype("category")
df["season"] = df["season"].astype("category")
df["reference_date"] = df["reference_date"].astype("category")

df["season_idx"] = df["season"].astype("category").cat.codes
df["model_idx"]  = df["model"].astype("category").cat.codes
df["ref_idx"]    = df["reference_date"].astype("category").cat.codes

season_idx = df["season_idx"].values
model_idx  = df["model_idx"].values
ref_idx    = df["ref_idx"].values
ED         = df["ED"].values.astype(int)
h          = df["training_horizon"].values
y          = df["log_rel_wis"].values

n_season = df["season_idx"].nunique()
n_model  = df["model_idx"].nunique()
n_ref    = df["ref_idx"].nunique()


# build pyMC model
with pm.Model() as learning_model:

    # parameters
    ## noise
    sigma = pm.HalfNormal("sigma", 0.5)                         # Observation noise
    sigma_ref = pm.HalfNormal("sigma_ref", 0.5)                 # Reference-date random noise
    b_ref = pm.Normal("b_ref", 0.0, sigma_ref, shape=n_ref)     # Reference-date random intercept
    ## asymptote \mu
    mu_global = pm.Normal("mu_global", -1.0, 1/3)
    mu_ED = pm.Normal("mu_ED", 0, 1/3)
    mu_season = pm.Normal("mu_season", 0.0, 1/3, shape=n_season)
    mu_model = pm.Normal("mu_model", 0.0, 1/3, shape=n_model)
    mu = mu_global + mu_ED + mu_season[season_idx] + mu_model[model_idx]
    # maximum learning gain \Delta
    Delta_model = pm.Normal("Delta_model", -1, 1/3, shape=n_model)
    Delta_season = pm.Normal("Delta_season", 0.0, 1/3, shape=n_season)
    Delta_ED = pm.Normal("Delta_ED", 0.0, 1/3)
    Delta = (
        Delta_model[model_idx]
        + Delta_season[season_idx]
        + Delta_ED * ED
    )
    delta_pos = pm.Deterministic("delta_pos", pt.exp(Delta))
    ## learning rate \kappa
    kappa_model = pm.Normal("kappa_model", -1.0, 1/3, shape=n_model)
    kappa_season = pm.Normal("kappa_season", 0.0, 1/3, shape=n_season)
    kappa_ED = pm.Normal("kappa_ED", 0.0, 1/3)
    kappa = (
        kappa_model[model_idx]
        + kappa_season[season_idx]
        + kappa_ED * ED
    )
    kappa_pos = pm.Deterministic("kappa_pos", pt.exp(kappa))

    # Mean learning curve
    mu_hat = (
        mu
        + delta_pos * pt.exp(-kappa_pos * h)
        + b_ref[ref_idx]
    )

    # Likelihood
    y_obs = pm.Normal(
        "y_obs",
        mu=mu_hat,
        sigma=sigma,
        observed=y
    )

# sample model
with learning_model:
    trace = pm.sample(
        250,
        tune=250,
        chains=4
    )

# sample posterior predictive
with learning_model:
    ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"])

# Generate traces
variables2plot = [
    "sigma", "sigma_ref",
    "mu_global", "mu_ED", "mu_season", "mu_model",
    "Delta_model", "Delta_season", "Delta_ED",
    "kappa_model", "kappa_season", "kappa_ED"
    ]
os.makedirs('trace', exist_ok=True)
for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(f'trace/trace-{var}.pdf')
    plt.close()


# Posterior summaries
from scipy.stats import skew
def summarize_posterior(trace, var_names):
    """
    Summarize posterior samples for PyMC variables.
    Handles both scalar and array variables automatically.
    
    Returns a DataFrame with:
    Parameter | Mean | SD | p-value | Skew
    """
    rows = []
    
    for var in var_names:
        # select variable from trace
        try:
            samples = trace.posterior[var].stack(draws=("chain", "draw")).values
        except KeyError:
            print(f"Variable {var} not found in trace. Skipping.")
            continue

        # if array (multi-dimensional), loop over axes
        if samples.ndim > 1:
            for idx in np.ndindex(samples.shape[:1]):  # skip draws axis
                # flatten the draws
                flat_samples = samples[idx, :].flatten()
                median_val = np.median(flat_samples)
                mean_val = flat_samples.mean()
                sd_val = flat_samples.std()
                skew_val = skew(flat_samples)
                p_val = 2 * min((flat_samples < 0).mean(), (flat_samples > 0).mean())
                
                param_name = f"{var}{idx}" if len(idx) > 1 else f"{var}[{idx[0]}]"
                rows.append({
                    "Parameter": param_name,
                    "Median": np.round(median_val, 3),
                    "Mean": np.round(mean_val, 3),
                    "SD": np.round(sd_val, 3),
                    "p-value": np.round(p_val, 3),
                    "Skew": np.round(skew_val, 3)
                })
        else:
            # scalar
            flat_samples = samples.flatten()
            median_val = np.median(flat_samples)
            mean_val = flat_samples.mean()
            sd_val = flat_samples.std()
            skew_val = skew(flat_samples)
            p_val = 2 * min((flat_samples < 0).mean(), (flat_samples > 0).mean())
            rows.append({
                "Parameter": var,
                "Median": np.round(median_val, 3),
                "Mean": np.round(mean_val,3),
                "SD": np.round(sd_val,3),
                "p-value": np.round(p_val,3),
                "Skew": np.round(skew_val,3),
            })
    
    return pd.DataFrame(rows)

summary_df = summarize_posterior(trace, variables2plot)
print(summary_df)


# visualise goodness-of-fit
y_hat_samples = ppc.posterior_predictive["y_obs"].values
n_chain, n_draw, n_data = y_hat_samples.shape
y_hat_log = y_hat_samples.reshape(n_chain * n_draw, n_data)

# build grouping
group_cols = ["model", "season", "training_horizon", "ED"]
df_groups = (
    df[group_cols]
    .reset_index(drop=True)
)

# compute modeled geometric mean per group
results = []
for g_vals, idx in df_groups.groupby(group_cols).groups.items():
    # idx = indices of df rows in this group
    idx = np.array(list(idx))

    # mean over observations, then mean over posterior draws
    mean_log_wis = y_hat_log[:, idx].mean(axis=1).mean()

    results.append({
        "model": g_vals[0],
        "season": g_vals[1],
        "training_horizon": g_vals[2],
        "ED": g_vals[3],
        "modeled_geo": np.exp(mean_log_wis)
    })

modeled_grouped = pd.DataFrame(results)

# paired bootstrap of empirical values
def geo_mean(x):
    return np.exp(np.mean(np.log(x)))

empirical_grouped = (
    df.groupby(["model", "season", "training_horizon", "ED"])["relative_WIS_nodrift"]
      .apply(geo_mean)
      .reset_index(name="empirical_geo")
)

# merge plotting dataframe
plot_df = empirical_grouped.merge(
    modeled_grouped,
    on=["model", "season", "training_horizon", "ED"],
    how="left"
)

# aesthetics
models = plot_df["model"].unique()
seasons = plot_df["season"].unique()
import matplotlib.pyplot as plt
colors = dict(zip(models, [plt.cm.tab10.colors[i] for i in [1, 2, 0]]))
markers = dict(zip(models, ["o", "^", "s", "D", "v"]))

# plot

# ensure consistent ordering
seasons = sorted(seasons)
ED_levels = [0, 1]

n_season = len(seasons)
n_ED = len(ED_levels)

fig, axes = plt.subplots(
    n_ED, n_season,
    figsize=(4 * n_season, 3.2 * n_ED),
    sharex=True,
    sharey=True
)

# Make axes 2D even if one dimension is 1
if n_ED == 1:
    axes = np.array([axes])
if n_season == 1:
    axes = axes[:, np.newaxis]

for i, ED_val in enumerate(ED_levels):
    for j, season in enumerate(seasons):
        ax = axes[i, j]

        df_panel = plot_df[
            (plot_df["ED"] == ED_val) &
            (plot_df["season"] == season)
        ]

        for model in models:
            df_m = df_panel[df_panel["model"] == model]

            # modeled line
            ax.plot(
                df_m["training_horizon"],
                df_m["modeled_geo"],
                color=colors[model],
                linewidth=1
            )

            # empirical points
            ax.scatter(
                df_m["training_horizon"],
                df_m["empirical_geo"],
                color=colors[model],
                marker=markers[model],
                s=30,
                zorder=3,
                label=model if (i == 0 and j == 0) else None
            )

        # titles & labels
        if i == 0:
            ax.set_title(season)

        if j == 0:
            ax.set_ylabel(
                "Rel. WIS (sGRW)\n"
                + ("ED visits" if ED_val == 1 else "No ED visits")
            )

        if i == n_ED - 1:
            ax.set_xlabel("Number of training seasons")

        ax.set_ylim(0.30, 0.85)
        ax.grid(False)

# shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=len(models),
    frameon=False
)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.show()
