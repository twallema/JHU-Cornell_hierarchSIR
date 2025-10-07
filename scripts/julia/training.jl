using Turing
using JLD2
using BenchmarkTools
using ForwardDiff
using SciMLSensitivity
using ReverseDiff

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using HierarchicalSIR

run_case = -1
if length(ARGS) >= 1
    run_case += 1
    if contains(ARGS[1], "emcee")
        run_case += 1
    elseif !contains(ARGS[1], "nuts")
        error("First argument must be either 'emcee' or 'nuts'")
    end
    if length(ARGS) >= 2 && contains(ARGS[2], "small")
        run_case += 2
    end
    if length(ARGS) >= 3 && contains(ARGS[3], "rev")
        run_case += 4
    end
end

config = AnalysisConfig()
data, population, t_span, season2idx = data_pipeline(config)

small_model = hierarchical_SIR_wo_bounds(data, population, t_span; n_Δβ = 6)
full_model = hierarchical_SIR(data, population, t_span; n_Δβ = 12)

small_init = get_initial_guess(small_model, config.seasons)
full_init = get_initial_guess(full_model, config.seasons)

####################
# Benchmarking code
####################
function bench_emcee(model, init_params, N)
    sample(model, Emcee(2*length(init_params)), N; init_params = last.(init_params), progress=false)
end

function bench_nuts_default(model, init_params, N)
    sample(model, NUTS(), N; init_params = last.(init_params), progress=false)
end

function bench_nuts_forward(model, init_params, N)
    sample(model, NUTS(; adtype=AutoForwardDiff()), N; init_params = last.(init_params), progress=false)
end

function bench_nuts_reverse(model, init_params, N)
    sample(model, NUTS(adtype=AutoReverseDiff()), N; init_params = last.(init_params), progress=false)
end

println("Running with case $(run_case)")

if run_case < 0
    # Run benchmarks
    N_benchmark_inner = 100 # samples per chain
    N_benchmark_outer = 5  # number of chains
    seconds = 600

    println("Benchmarking sampling methods:")

    emcee_benchmark = @benchmark bench_emcee($small_model, $small_init, $N_benchmark_inner) samples=N_benchmark_outer seconds=seconds
    println("Emcee: ", emcee_benchmark)

    nuts_default_benchmark = @benchmark bench_nuts_default($small_model, $small_init, $N_benchmark_inner) samples=N_benchmark_outer seconds=seconds
    println("NUTS (default): ", nuts_default_benchmark)

    nuts_forward_benchmark = @benchmark bench_nuts_forward($small_model, $small_init, $N_benchmark_inner) samples=N_benchmark_outer seconds=seconds
    println("NUTS (ForwardDiff): ", nuts_forward_benchmark)

    nuts_reverse_benchmark = @benchmark bench_nuts_reverse($small_model, $small_init, $N_benchmark_inner) samples=N_benchmark_outer seconds=seconds
    println("NUTS (ReverseDiff): ", nuts_reverse_benchmark)

    exit()
end

##################
# Actual sampling
##################

chain_dir = normpath(joinpath(@__DIR__, "..", "..", "data", "julia_chains"))
isdir(chain_dir) || mkpath(chain_dir)

function run_with_cases(run_case, full_model, small_model, full_init, small_init)
    model = run_case >= 2 ? small_model : full_model
    model_str = run_case >= 2 ? "small" : "full"
    init_params = run_case >= 2 ? small_init : full_init

    if run_case % 2 == 0
        adtype = run_case >= 4 ? AutoReverseDiff() : AutoForwardDiff()
        ad_type_str = run_case >= 4 ? "rev" : "fwd"
        println("Running NUTS with $(adtype) on $model_str")
        chain = sample(model, NUTS(; adtype), MCMCThreads(), 2500, 6; init_params = last.(init_params))
        JLD2.@save joinpath(chain_dir, "n_$(model_str)_$(ad_type_str).jld2") chain
    else
        println("Running Emcee on $model_str")
        n_walkers = 2 * length(init_params)
        e_small_init = sample(small_model, Emcee(n_walkers), 10000; init_params = last.(init_params))
        JLD2.@save joinpath(chain_dir, "e_$(model_str).jld2") e_small_init
    end
end

@time run_with_cases(run_case, full_model, small_model, full_init, small_init)
