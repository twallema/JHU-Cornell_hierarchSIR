using Turing
using JLD2
using BenchmarkTools
using ForwardDiff
using SciMLSensitivity
using ReverseDiff

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using HierarchicalSIR

config = AnalysisConfig()
data, population, t_span, season2idx = data_pipeline(config)

small_model = hierarchical_SIR_wo_bounds(data, population, t_span; n_Δβ = config.n_Δβ)
full_model = hierarchical_SIR(data, population, t_span; n_Δβ = config.n_Δβ)

small_init = get_initial_guess(small_model, config.seasons)
full_init = get_initial_guess(full_model, config.seasons)

####################
# Benchmarking code
####################

N_benchmark = 250

function bench_emcee(model, init_params, N)
    sample(model, Emcee(2*length(init_params)), N; init_params = last.(init_params))
end

function bench_nuts_default(model, init_params, N)
    sample(model, NUTS(), N; init_params = last.(init_params))
end

function bench_nuts_forward(model, init_params, N)
    sample(model, NUTS(; adtype=AutoForwardDiff()), N; init_params = last.(init_params))
end

function bench_nuts_reverse(model, init_params, N)
    sample(model, NUTS(adtype=AutoReverseDiff()), N; init_params = last.(init_params))
end

# Run benchmarks
println("Benchmarking sampling methods:")

emcee_benchmark = @benchmark bench_emcee($small_model, $small_init, $N_benchmark)
println("Emcee: ", emcee_benchmark)

nuts_default_benchmark = @benchmark bench_nuts_default($small_model, $small_init, $N_benchmark)
println("NUTS (default): ", nuts_default_benchmark)

nuts_forward_benchmark = @benchmark bench_nuts_forward($small_model, $small_init, $N_benchmark)
println("NUTS (ForwardDiff): ", nuts_forward_benchmark)

nuts_reverse_benchmark = @benchmark bench_nuts_reverse($small_model, $small_init, $N_benchmark)
println("NUTS (ReverseDiff): ", nuts_reverse_benchmark)

##################
# Actual sampling
##################

chain_dir = normpath(joinpath(@__DIR__, "..", "..", "data", "julia_chains" ))
isdir(chain_dir) || mkpath(chain_dir)

e_small_init = sample(small_model, Emcee(150), 10000; init_params = last.(small_init))
JLD2.@save joinpath(chain_dir, "e_small_init.jld2") e_small_init

n_small_init = sample(small_model, NUTS(adtype=AutoForwardDiff()), MCMCThreads(), 2500, 6; init_params = last.(small_init))
JLD2.@save joinpath(chain_dir, "n_small_init.jld2") n_small_init

e_full_init = sample(full_model, Emcee(150), 10000; init_params = last.(full_init))
JLD2.@save joinpath(chain_dir, "e_full_init.jld2") e_full_init

