"""
    create_SIR(max_T, n_Δβ)

Construct the in-place right-hand side for the hierarchical SIR system. `max_T`
sets the time horizon for the piecewise-linear transmission adjustment with
`n_Δβ` knots.
"""
function create_SIR(max_T, n_Δβ)
    nodes = range(0, max_T, length=n_Δβ)
    linear_int = (t, Δβ) -> begin
        idx = clamp(searchsortedlast(nodes, t), 1, length(Δβ) - 1)
        t1, t2 = nodes[idx], nodes[idx + 1]
        w = (t - t1) / (t2 - t1)
        return Δβ[idx] * (1 - w) + Δβ[idx + 1] * w
    end

    return function sir_rhs!(du, u, p, t)
        S, I, R, I_inc, H_inc_LCT0, H_inc_LCT1, H_inc_LCT2, H_inc_LCT3, H_inc_LCT4, H_inc = u
        ρᵢ = p[1]
        Tₕ = p[2]
        ρₕ = p[3]
        β = p[4]
        γ = p[5]
        P = 5.0 / Tₕ

        Δβ = p[6:end]
        modifier = linear_int(t, Δβ)
        λ = β * (1 + modifier) * S * I / (S + I + R)

        du[1] = -λ
        du[2] = λ - γ * I
        du[3] = γ * I
        du[4] = ρᵢ * λ - I_inc
        du[5] = ρₕ * λ - P * H_inc_LCT0
        du[6] = P * H_inc_LCT0 - P * H_inc_LCT1
        du[7] = P * H_inc_LCT1 - P * H_inc_LCT2
        du[8] = P * H_inc_LCT2 - P * H_inc_LCT3
        du[9] = P * H_inc_LCT3 - P * H_inc_LCT4
        du[10] = P * H_inc_LCT4 - H_inc
    end
end

"""
    unpack_value(solution, idx, comp)

Helper to access component `comp` at observation index `idx` from a solved
trajectory.
"""
@inline unpack_value(v::ODESolution, t, obs) = v[t][obs]
@inline unpack_value(v::Vector{<:Vector}, t, obs) = v[t][obs]

"""
    hierarchical_SIR_wo_bounds(data, population, t_span; kwargs...)

Hierarchical Turing model without hard parameter bounds. Fits the SIR system to
`data` (incidence and hospitalizations) over `t_span` for a population size.
"""
@model function hierarchical_SIR_wo_bounds(data, population, t_span; n_Δβ=7, dt=7.0, γ=Γ, template=ODEProblem(create_SIR(t_span[2], n_Δβ), zeros(10), t_span, zeros(5+n_Δβ)))
    n_seasons = size(data, 2)

    ρᵢ ~ Beta(8.19, 293.29)
    Tₕ ~ LogNormal(log(1.322936585), 0.337142555)
    ρₕ ~ Beta(10.921, 3103.543)
    fᵢ ~ Beta(2.3172, 1218.)
    fᵣ ~ Beta(43.659, 99.066)
    T = eltype(ρᵢ)

    β ~ filldist(Beta(6, 7.5), n_seasons)
    Δβ_raw ~ filldist(Beta(5, 5), n_Δβ)
    Δβ = T.(2 .* (Δβ_raw .- 0.5))

    cb = PositiveDomain(save = false)
    eval = prob -> solve(prob, Tsit5(); callback=cb, saveat=dt, save_idxs=[4, 10], verbose=false)

    x0 = MVector{10,T}(one(T) - (T(fᵢ) + T(fᵣ)), T(fᵢ), T(fᵣ), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T)) .* T(population)

    p = MVector{5 + n_Δβ, T}(undef)
    p[1] = ρᵢ
    p[2] = Tₕ
    p[3] = ρₕ
    p[4] = zero(T)
    p[5] = γ
    @views p[6:end] .= Δβ

    for season in 1:n_seasons
        p[4] = β[season]
        c_prob = remake(template; u0 = x0, p = p)
        sol = eval(c_prob)
        obs = sol

        if sol.retcode != ReturnCode.Success
            Turing.@addlogprob!(-Inf)
            data[:, season, 1] ~ Normal(0, 1)
            data[:, season, 2] ~ Normal(0, 1)
        else
            for i in axes(data, 1)
                d1 = unpack_value(obs, i, 1)
                d2 = unpack_value(obs, i, 2)
                data[i, season, 1] ~ Normal(d1, d1 + 0.5)
                data[i, season, 2] ~ Normal(d2, d2 + 0.5)
            end
        end
    end
end

"""
    hierarchical_SIR(data, population, t_span; kwargs...)

Bounded hierarchical Turing model with truncated priors for season-level and
hyper parameters.
"""
@model function hierarchical_SIR(data, population, t_span; n_Δβ=12, dt=7.0, γ=Γ, template=ODEProblem(create_SIR(t_span[2], n_Δβ), zeros(10), t_span, zeros(5+n_Δβ)))
    n_seasons = size(data, 2)

    βμ ~ truncated(Normal(0.455, 0.25), 0.05, 0.95)
    βσ ~ Exponential(0.25)
    T = eltype(βμ)

    Δβμ ~ filldist(truncated(Laplace(0.0, 0.33), -1.0, 1.0), n_Δβ)
    Δβσ ~ filldist(truncated(Exponential(0.15), 0, 1), n_Δβ)

    ρᵢ ~ filldist(truncated(LogNormal(log(0.025679272), 0.334315924), 1e-3, 0.075), n_seasons)
    Tₕ ~ filldist(truncated(LogNormal(log(1.322936585), 0.337142555), 0.5, 14), n_seasons)
    ρₕ ~ filldist(truncated(LogNormal(log(0.003356751), 0.295460858), 1e-4, 0.0075), n_seasons)
    fᵢ ~ filldist(truncated(LogNormal(log(0.00018612), 0.205517672), 1e-6, 0.001), n_seasons)
    fᵣ ~ filldist(truncated(Normal(0.303313396, 0.03520003), 0.10, 0.90), n_seasons)

    β = Vector{T}(undef, n_seasons)
    Δβ = Array{T}(undef, n_seasons, n_Δβ)

    x0 = MVector{10,T}(undef)
    p = MVector{5 + n_Δβ, T}(undef)
    p[5] = T(γ)

    cb = PositiveDomain(save = false)
    eval = prob -> solve(prob, Tsit5(); callback=cb, saveat=dt, save_idxs=[4, 10], verbose=false)

    for season in 1:n_seasons
        β[season] ~ truncated(Normal(βμ, βσ), 0.05, 0.95)
        Δβ[season, :] ~ arraydist(truncated.(Normal.(Δβμ, Δβσ), -1.0, 1.0))

        p[1] = ρᵢ[season]
        p[2] = Tₕ[season]
        p[3] = ρₕ[season]
        p[4] = β[season]
        p[6:end] .= Δβ[season, :]

        x0[2] = fᵢ[season]
        x0[3] = fᵣ[season]
        x0[1] = one(T) - (x0[2] + x0[3])
        x0[4:end] .= zero(T)
        x0 .*= T(population)

        c_prob = remake(template; u0 = x0, p = p)
        sol = eval(c_prob)

        if sol.retcode != ReturnCode.Success
            @warn "failed with $p and $(x0[1:3])"
            Turing.@addlogprob!(-Inf)
            data[:, season, 1] ~ Normal(0, 1)
            data[:, season, 2] ~ Normal(0, 1)
        else
            for i in axes(data, 1)
                d1 = unpack_value(sol, i, 1)
                d2 = unpack_value(sol, i, 2)
                data[i, season, 1] ~ Normal(d1, d1 + 0.5)
                data[i, season, 2] ~ Normal(d2, d2 + 0.5)
            end
        end
    end
end

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
