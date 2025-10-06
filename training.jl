using Turing
using JLD2
using ForwardDiff
using SciMLSensitivity
using ReverseDiff

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using Revise
using HierarchicalSIR

config = AnalysisConfig()
data, population, t_span, season2idx = data_pipeline(config)

small_model = hierarchical_SIR_wo_bounds(data, population, t_span; n_Δβ = config.n_Δβ)
full_model = hierarchical_SIR(data, population, t_span; n_Δβ = config.n_Δβ)

small_init= get_initial_guess(small_model, config.seasons)
full_init = get_initial_guess(full_model, config.seasons)

e_chain_small = sample(small_model, Emcee(30), 25; init_params = last.(small_init))
describe(e_chain_small)
JLD2.@save "test_chain_emcee_small_init.jld2" e_chain_small

n_chain_small = sample(small_model, NUTS(adtype=AutoForwardDiff()), MCMCThreads(), 500, 4; init_params = last.(small_init))
describe(n_chain_small)
JLD2.@save "test_chain_nuts_small_init.jld2" n_chain_small

e_chain_full = sample(full_model, Emcee(128), 500)
describe(e_chain_full)
JLD2.@save "test_chain_emcee_full_init.jld2" e_chain_full

n_chain_full = sample(full_model, NUTS(adtype=AutoForwardDiff()), MCMCThreads(), 500, 4)
describe(n_chain_full)
JLD2.@save "test_chain_nuts_full_init.jld2" n_chain_full
