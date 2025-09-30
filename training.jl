using Dates
using Random
using Turing
using JLD2
using DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using HierarchicalSIR

Base.@kwdef struct AnalysisConfig
    seasons::Vector{String} =  ["2014-2015", "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",]
    identifier::String = "exclude_None"
    start_month::Int = 10
    end_month::Int = 5
    n_Δβ::Int = 7
    population_fips::Int = 37
    seed::Int = 123
    nuts_samples::Int = 1000
    nuts_chains::Int = Threads.nthreads()
    emcee_walkers::Int = 30
    emcee_steps::Int = 8000
end

function prepare_calibration_window(seasons::Vector{String}, start_month::Int, end_month::Int)
    start_dates = [Date(parse(Int, season[1:4]), start_month, 1) for season in seasons]
    end_dates = [Date(parse(Int, season[1:4]) + 1, end_month, 1) for season in seasons]
    return start_dates, end_dates
end

config = AnalysisConfig()

start_dates, end_dates = prepare_calibration_window(config.seasons, config.start_month, config.end_month)
datasets, I_dataset, H_dataset, season2idx = concatenate_datasets(config.seasons, start_dates, end_dates)
data = cat(Matrix(select(I_dataset, Not(:week))), Matrix(select(H_dataset, Not(:week))), dims = 3)
t_span = (0.0, 7.0 * (size(data, 1) - 1))
population = get_demography(config.population_fips)

model = hierarchical_SIR_wo_bounds(data, population, t_span; n_Δβ = config.n_Δβ)
chain = sample(model, NUTS(adtype=AutoForwardDiff()), MCMCThreads(), 1000, 3)

describe(chain)

using JLD2
@save "test_chain.jld2" chain config

using Plots, StatsPlots
plot(chain)
