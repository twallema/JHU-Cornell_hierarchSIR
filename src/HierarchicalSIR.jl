module HierarchicalSIR

using CSV
using DataFrames
using Dates
using LinearAlgebra
using DifferentialEquations
using Distributions
using MCMCChains
using Random
using StaticArrays
using Statistics
using Turing

const Γ = 1 / 3.5
const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data"))

include("util.jl")
include("hierarchicalSIR.jl")

export Γ,
       create_SIR,
       hierarchical_SIR,
       hierarchical_SIR_wo_bounds,
       concatenate_datasets,
       simulate,
       unpack_and_simulate,
       simulate_posterior,
       summarize_posterior,
       unpack_value,
       get_demography,
       get_NC_influenza_data,
       get_initial_guess,
       select_parameters,
       convert_seasonal_parameters,
       convert_hyper_parameters

end
