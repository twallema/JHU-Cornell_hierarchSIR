
"""
    simulate(ρᵢ, Tₕ, ρₕ, fᵢ, fᵣ, β, Δβ, population, t_span; kwargs...)

Simulate the deterministic SIR dynamics for a single season. Returns an
`ODESolution`; keyword arguments are forwarded to `DifferentialEquations.solve`.
"""
function simulate(ρᵢ, Tₕ, ρₕ, fᵢ, fᵣ, β, Δβ, population, t_span; kwargs...)
    p = [ρᵢ, Tₕ, ρₕ, β, Γ, Δβ...]

    x0 = zeros(10)
    x0[2] = fᵢ
    x0[3] = fᵣ
    x0[1] = 1 - (x0[2] + x0[3])
    x0 .*= population

    prob = ODEProblem(create_SIR(t_span[2], length(Δβ)), x0, t_span, p)
    return solve(prob, Tsit5(); kwargs...)
end

@inline season_symbol = (parameter, season) -> Symbol("$(parameter)[$(season)]")

"""
    unpack_and_simulate(chain, population, t_span; season=1, kwargs...)

Simulate a trajectory using an aggregate over a posterior chain. The default
aggregate is `mean`.
"""
function unpack_and_simulate(chain::MCMCChains.Chains, population, t_span; agg_f=mean, season=1, kwargs...)
    param_names = MCMCChains.names(chain)

    ρᵢ = :ρᵢ in param_names ? agg_f(chain[:ρᵢ]) : agg_f(chain[season_symbol("ρᵢ", season)])
    Tₕ = :Tₕ in param_names ? agg_f(chain[:Tₕ]) : agg_f(chain[season_symbol("Tₕ", season)])
    ρₕ = :ρₕ in param_names ? agg_f(chain[:ρₕ]) : agg_f(chain[season_symbol("ρₕ", season)])
    fᵢ = :fᵢ in param_names ? agg_f(chain[:fᵢ]) : agg_f(chain[season_symbol("fᵢ", season)])
    fᵣ = :fᵣ in param_names ? agg_f(chain[:fᵣ]) : agg_f(chain[season_symbol("fᵣ", season)])
    β = :β in param_names ? agg_f(chain[:β]) : agg_f(chain[season_symbol("β", season)])

    Δβ_keys = [k for k in param_names if occursin("Δβ", String(k))]
    if occursin("raw", String(Δβ_keys[1]))
        Δβ_raw = [agg_f(chain[k]) for k in sort(Δβ_keys)]
        Δβ = 2 .* (Δβ_raw .- 0.5)
    else
        season_keys = [k for k in Δβ_keys if occursin("[$season,", String(k))]
        Δβ = [agg_f(chain[k]) for k in sort(season_keys)]
    end

    return simulate(ρᵢ, Tₕ, ρₕ, fᵢ, fᵣ, β, Δβ, population, t_span; kwargs...)
end

"""
    simulate_posterior_summary(chain, population, t_span; kwargs...)

Simulate trajectories for posterior samples and compute summary statistics.
Additional keyword arguments are forwarded to `simulate`.

# Keyword arguments
- `season`: season index (1-based) to evaluate.
- `nsamples`: number of samples drawn without replacement (defaults to all).
- `quantiles`: iterable of probabilities for which quantiles are reported.
- `ci_level`: central confidence level (e.g. 0.95); set to `nothing` to skip.
"""
function simulate_posterior(chain::MCMCChains.Chains, population, t_span;
    season::Int = 1,
    n_selected::Union{Nothing,Int} = nothing,
    save_idxs=[4, 10],
    saveat=7.0,
    kwargs...
)
    n_samples = Int64(length(chain.value[:]) / size(chain, 2))
    selected_idxs = isnothing(n_selected) ? Base.OneTo(n_samples) : randperm(n_samples)[1:n_selected]
    n_selected = length(selected_idxs)

    stack = nothing
    time_grid = nothing

    for (i, idx) in enumerate(selected_idxs)
        selector = x -> vec(x)[idx]
        sol = unpack_and_simulate(chain, population, t_span; season, agg_f=selector, save_idxs, saveat, kwargs...)

        time_grid === nothing && (time_grid = collect(sol.t))
        sol_array = Array(sol)
        if stack === nothing
            n_states, n_times = size(sol_array)
            stack = Array{eltype(sol_array)}(undef, n_states, n_times, n_selected)
        else
            if size(sol_array, 1) != size(stack, 1) || size(sol_array, 2) != size(stack, 2)
                throw(ArgumentError("simulated trajectories do not share the same state/time dimensions"))
            end
        end
        stack[:, :, i] = sol_array
    end

    return time_grid, stack
end

function summarize_posterior_sample(stack::AbstractArray, quantiles::AbstractVector)
    mean_traj = dropdims(mean(stack; dims=3), dims=3)
    std_traj = dropdims(std(stack; dims=3), dims=3)

    quantile_stats = Dict{Float64, Matrix{eltype(stack)}}()
    for q in quantiles
        quantile_stats[Float64(q)] = vcat(
            [quantile(stack[1, t, :], q) for t in axes(stack, 2)]',
            [quantile(stack[2, t, :], q) for t in axes(stack, 2)]',
        )
    end

    return (
        mean = mean_traj,
        std = std_traj,
        quantiles = quantile_stats,
    )
end

function summarize_posterior(chain::MCMCChains.Chains, population, t_span;
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975],
    kwargs...
)
    time_grid, stack = simulate_posterior(chain, population, t_span; kwargs...)

    summary = summarize_posterior_sample(stack, quantiles)

    return (
        time = time_grid,
        mean = summary.mean,
        std = summary.std,
        quantiles = summary.quantiles,
    )
end
