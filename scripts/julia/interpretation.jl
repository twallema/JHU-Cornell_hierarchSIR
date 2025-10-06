using JLD2
using Plots
using StatsPlots
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using HierarchicalSIR

config = AnalysisConfig()

data, population, t_span, season2idx = data_pipeline(config)

JLD2.@load "data/julia_chains/test_chain_emcee_full.jld2"
chain = e_chain_full

season = 2
sol = unpack_and_simulate(chain, population, t_span; season, save_idxs=[4,10], agg_f=mean)
plot(sol)
ts = 0.0:7.0:7.0 * (size(data, 1) - 1)

data_I = data[:, season, 1]
data_H = data[:, season, 2]

res = summarize_posterior(chain, population, t_span; season, n_selected=1000)
plot(res.time, res.mean', label = "mean")
plot!(res.time, res.quantiles[0.75]', fillrange = res.quantiles[0.25]', fillalpha = 0.2, linealpha = 0, label = nothing)
plot!(res.time, res.quantiles[0.5]', label = "median")
scatter!(ts, data_I, label = "data_I")
scatter!(ts, data_H, label = "data_H")
