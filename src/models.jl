"""
    create_SIR(max_T, n_Δβ)

Construct the in-place right-hand side for the hierarchical SIR system. `max_T`
sets the time horizon for the piecewise-linear transmission adjustment with
`n_Δβ` knots.
"""
function create_SIR(max_T, n_Δβ)
    nodes = range(0, max_T, length=(n_Δβ+1))
    # linear_int = (t, Δβ) -> begin
    #     idx = clamp(searchsortedlast(nodes, t), 1, length(Δβ) - 1)
    #     t1, t2 = nodes[idx], nodes[idx + 1]
    #     w = (t - t1) / (t2 - t1)
    #     return Δβ[idx] * (1 - w) + Δβ[idx + 1] * w
    # end

    constant_int = (t, Δβ) -> begin
        idx = clamp(searchsortedlast(nodes, t), 1, length(Δβ))
        return Δβ[idx]
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
        modifier = constant_int(t, Δβ)
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
@inline unpack_value(v::AbstractMatrix, t, obs) = v[obs, t]

@inline has_succeeded(sol::ODESolution, n_obs, n_time_steps) = sol.retcode == ReturnCode.Success
@inline has_succeeded(sol::Vector{<:Vector}, n_obs, n_time_steps) = length(sol) == n_time_steps && length(sol[1]) == n_obs && all(x -> all(isfinite, x), sol)
@inline has_succeeded(sol::AbstractMatrix, n_obs, n_time_steps) = size(sol, 1) == n_obs && size(sol, 2) == n_time_steps && all(isfinite, sol)
@inline has_succeeded(sol::Nothing) = false

"""
    hierarchical_SIR_wo_bounds(data, population, t_span; kwargs...)

Hierarchical Turing model without hard parameter bounds. Fits the SIR system to
`data` (incidence and hospitalizations) over `t_span` for a population size.
"""
@model function hierarchical_SIR_wo_bounds(data, population, t_span; n_Δβ=6, dt=7.0, γ=Γ, template=ODEProblem(create_SIR(t_span[2], n_Δβ), zeros(10), t_span, zeros(5+n_Δβ)))
    n_seasons = size(data, 2)
    n_time_steps = size(data, 1)
    @assert n_time_steps == round(Int, (t_span[2] - t_span[1]) / dt) + 1 "data time dimension does not match t_span/dt"

    ρᵢ ~ Beta(8.19, 293.29)
    Tₕ ~ LogNormal(log(1.322936585), 0.337142555)
    ρₕ ~ Beta(10.921, 3103.543)
    fᵢ ~ Beta(2.3172, 1218.)
    fᵣ ~ Beta(43.659, 99.066)

    β ~ filldist(Beta(32, 40), n_seasons)

    # Hierarchical priors for Δβ parameters
    α_Δβ ~ filldist(Exponential(5), n_Δβ)  # Mean = 5, constrains α > 0
    β_Δβ ~ filldist(Exponential(5), n_Δβ)  # Mean = 5, constrains β > 0

    Δβ_raw = Array{eltype(α_Δβ)}(undef, n_seasons, n_Δβ)
    for season in 1:n_seasons
        Δβ_raw[season, :] ~ filldist(Beta(α_Δβ[season], β_Δβ[season]), n_Δβ)
    end
    Δβ = @. 2 * (Δβ_raw - 0.5)

    cb = PositiveDomain(save = false)
    eval = prob -> solve(prob, Tsit5(); callback=cb, saveat=dt, save_idxs=[4, 10], verbose=false)
    one_T = one(ρᵢ)
    zero_T = zero(ρᵢ)
    population_T = population * one_T

    # Construct AD-friendly initial conditions without instantiating tracked types manually.
    base_x0 = (
        one_T - (fᵢ + fᵣ),
        fᵢ,
        fᵣ,
        zero_T,
        zero_T,
        zero_T,
        zero_T,
        zero_T,
        zero_T,
        zero_T,
    )
    x0 = Vector{typeof(ρᵢ)}(undef, 10)
    @inbounds for (idx, val) in enumerate(base_x0)
        x0[idx] = val * population_T
    end

    p = Vector{typeof(ρᵢ)}(undef, 5 + n_Δβ)
    p[1] = ρᵢ
    p[2] = Tₕ
    p[3] = ρₕ
    p[4] = zero_T
    p[5] = γ * one_T

    for season in 1:n_seasons
        p[4] = β[season]
        p[6:end] .= Δβ[season, :]
        c_prob = remake(template; u0 = x0, p = p)
        sol = eval(c_prob)

        if !has_succeeded(sol, 2, n_time_steps)
            Turing.@addlogprob!(-Inf)
            data[:, season, 1] ~ Normal(0, 1)
            data[:, season, 2] ~ Normal(0, 1)
        else
            for i in 1:n_time_steps
                d1 = unpack_value(sol, i, 1)
                d2 = unpack_value(sol, i, 2)
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
    n_time_steps = size(data, 1)
    @assert n_time_steps == round(Int, (t_span[2] - t_span[1]) / dt) + 1 "data time dimension does not match t_span/dt"

    βμ ~ truncated(Normal(0.455, 0.055), 0.05, 0.95)
    βσ ~ Exponential(0.055)
    T = eltype(βμ)

    Δβμ ~ filldist(truncated(Normal(0.0, 0.1), -1.0, 1.0), n_Δβ)
    Δβσ ~ filldist(truncated(Exponential(0.15), 0, 1), n_Δβ)

    ρᵢ ~ filldist(truncated(LogNormal(log(0.025679272), 0.334315924), 1e-3, 0.075), n_seasons)
    Tₕ ~ filldist(truncated(LogNormal(log(1.322936585), 0.337142555), 0.5, 14), n_seasons)
    ρₕ ~ filldist(truncated(LogNormal(log(0.003356751), 0.295460858), 1e-4, 0.0075), n_seasons)
    fᵢ ~ filldist(truncated(LogNormal(log(0.00018612), 0.205517672), 1e-6, 0.001), n_seasons)
    fᵣ ~ filldist(truncated(Normal(0.33313396, 0.02320003), 0.10, 0.90), n_seasons)

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

        if !has_succeeded(sol, 2, n_time_steps)
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
