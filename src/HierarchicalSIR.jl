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
include("models.jl")
include("postprocess.jl")

export Γ,
       AnalysisConfig,
       data_pipeline,
       create_SIR,
       hierarchical_SIR,
       hierarchical_SIR_wo_bounds,
       simulate,
       unpack_and_simulate,
       simulate_posterior,
       summarize_posterior,
       get_initial_guess

end
