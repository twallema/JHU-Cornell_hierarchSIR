using JLD2
using Plots
using StatsPlots

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using Revise
using HierarchicalSIR

config = AnalysisConfig()

data, population, t_span, season2idx = data_pipeline(config)



density(e_chain_full[Symbol("β[1]")][:])
mean(e_chain_full[Symbol("β[1]")])
mode(e_chain_full[Symbol("β[1]")][:])
median(e_chain_full[Symbol("β[1]")])


season = 2
sol = unpack_and_simulate(e_chain_full, population, t_span; season, save_idxs=[4,10], agg_f=mean)
plot(sol)
ts = 0.0:7.0:7.0 * (size(data, 1) - 1)

data_I = data[:, season, 1]
data_H = data[:, season, 2]

res = summarize_posterior(e_chain_full, population, t_span; season, n_selected=5000)
plot(res.time, res.mean')
plot!(res.time, res.quantiles[0.5]', ribbon = (res.quantiles[0.75] .- res.quantiles[0.25])', alpha = 0.3)
scatter!(ts, data_I, label = "data_I")
scatter!(ts, data_H, label = "data_H")
