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
    modifier = isempty(Δβ) ? zero(typeof(β)) : linear_int(t, Δβ)
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

    ρᵢ ~ LogNormal(log(0.025679272), 0.334315924)
    Tₕ ~ LogNormal(log(1.322936585), 0.337142555)
    ρₕ ~ LogNormal(log(0.003356751), 0.295460858)
    fᵢ ~ LogNormal(log(0.00018612), 0.205517672)
    fᵣ ~ LogNormal(log(0.303313396), 0.12520003)
    T = eltype(ρᵢ)

    β ~ filldist(Beta(6, 7.5), n_seasons)
    Δβ_raw ~ filldist(Beta(5, 5), n_Δβ)
    Δβ = T.(2 .* (Δβ_raw .- 0.5))

    cb = PositiveDomain(save = false)
    eval = prob -> solve(prob, Tsit5(); callback=cb, saveat=dt, save_idxs=[4, 10], verbose=false)

    x0 = zeros(T, 10)
    x0[2] = T(fᵢ)
    x0[3] = T(fᵣ)
    x0[1] = one(T) - (x0[2] + x0[3])
    x0[4:end] .= zero(T)
    x0 .*= T(population)

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
@model function hierarchical_SIR(data, population, t_span; n_Δβ=7, dt=7.0, γ=Γ, template=ODEProblem(create_SIR(t_span[2], n_Δβ), zeros(10), t_span, zeros(5+n_Δβ)))
    n_seasons = size(data, 2)

    βμ ~ truncated(Normal(0.455, 0.25), 0.05, 0.95)
    βσ ~ Exponential(0.25)
    T = eltype(βμ)

    Δβμ ~ truncated(filldist(Laplace(0.0, 0.33), n_Δβ), -1.0, 1.0)
    Δβσ ~ filldist(Exponential(0.15), n_Δβ)

    ρᵢ ~ filldist(truncated(LogNormal(log(0.025679272), 0.334315924), 1e-3, 0.075), n_seasons)
    Tₕ ~ filldist(truncated(LogNormal(log(1.322936585), 0.337142555), 0.5, 14), n_seasons)
    ρₕ ~ filldist(truncated(LogNormal(log(0.003356751), 0.295460858), 1e-4, 0.0075), n_seasons)
    fᵢ ~ filldist(truncated(LogNormal(log(0.00018612), 0.205517672), 1e-6, 0.001), n_seasons)
    fᵣ ~ filldist(truncated(Normal(0.303313396, 0.03520003), 0.10, 0.90), n_seasons)

    β = Vector{T}(undef, n_seasons)
    Δβ = Array{T}(undef, n_seasons, n_Δβ)

    x0 = zeros(T, 10)
    p = MVector{5 + n_Δβ, T}(undef)
    p[5] = T(Γ)

    cb = PositiveDomain(save = false)
    eval = prob -> solve(prob, Tsit5(); callback=cb, saveat=dt, save_idxs=[4, 10], verbose=true)

    Σ = Diagonal((Δβσ) .^ 2)
    for season in 1:n_seasons
        β[season] ~ truncated(Normal(βμ, βσ), 0.05, 0.95)
        Δβ[season, :] ~ MvNormal(Δβμ, Σ)

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

"""
    unpack_and_simulate(sample, population, t_span; kwargs...)

Average posterior samples `sample` and run a deterministic simulation using the
mean parameters.
"""
function unpack_and_simulate(sample, population, t_span; kwargs...)
    return simulate(
        mean(sample[:ρᵢ]),
        mean(sample[:Tₕ]),
        mean(sample[:ρₕ]),
        mean(sample[:fᵢ]),
        mean(sample[:fᵣ]),
        mean(sample[Symbol("β[1]")]),
        [mean(sample[Symbol("Δβ[1, :][$(i)]")]) for i in 1:4],
        population, t_span;
        kwargs...
    )
end