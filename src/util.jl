"""
    data_path(parts...)

Join `parts` onto the project data directory.
"""
function data_path(parts...)
    return normpath(joinpath(DATA_DIR, parts...))
end

Base.@kwdef struct AnalysisConfig
    seasons::Vector{String} =  ["2014-2015", "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",]
    identifier::String = "exclude_None"
    start_month::Int = 10
    end_month::Int = 5
    n_Δβ::Int = 12
    population_fips::Int = 37
    seed::Int = 123
end

function prepare_calibration_window(seasons::Vector{String}, start_month::Int, end_month::Int)
    start_dates = [Date(parse(Int, season[1:4]), start_month, 1) for season in seasons]
    end_dates = [Date(parse(Int, season[1:4]) + 1, end_month, 1) for season in seasons]
    return start_dates, end_dates
end

"""
    get_demography(fips)

Return the population for the state identified by `fips`.
"""
function get_demography(fips::Int)
    demographies = CSV.read(data_path("interim", "demography", "demography.csv"), DataFrame)
    return filter(row -> row.fips_state == fips, demographies).population[1]
end

"""
    get_cdc_week_saturday(year, week)
    get_cdc_week_saturday(date)

Return the CDC surveillance week Saturday for the given ISO week or date.
"""
function get_cdc_week_saturday(year::Int, week::Int)
    jan4 = Date(year, 1, 4)
    week_start = jan4 - Day(Dates.dayofweek(jan4) - 1)
    return week_start + Week(week - 1) + Day(5)
end

function get_cdc_week_saturday(date::TimeType)
    year = Dates.year(date)
    week = Dates.week(date)
    return get_cdc_week_saturday(year, week)
end

"""
    load_hospitalization_and_ed_data(startdate, enddate)

Load and preprocess hospitalization and ED visit counts between the supplied
dates.
"""
function load_hospitalization_and_ed_data(startdate::Date, enddate::Date)
    hosp_path = data_path("raw", "cases", "hosp-admissions_NC_2010-2025.csv")
    ed_path = data_path("raw", "cases", "ED-visits_NC_2010-2025.csv")

    df_hosp = CSV.read(hosp_path, DataFrame, dateformat="yyyy-mm-dd")
    df_ed = CSV.read(ed_path, DataFrame, dateformat="yyyy-mm-dd")

    df_hosp.date = Date.(df_hosp[:, 1])
    df_ed.date = Date.(df_ed[:, 1])

    df_hosp.H_inc = df_hosp.flu_hosp ./ 7
    df_ed.I_inc = df_ed.flu_ED ./ 7

    select!(df_hosp, [:date, :H_inc])
    select!(df_ed, [:date, :I_inc])

    df_raw = outerjoin(df_hosp, df_ed, on=:date)
    filter!(row -> startdate ≤ row.date ≤ enddate, df_raw)
    df_raw.date = get_cdc_week_saturday.(df_raw.date)

    return df_raw
end

"""
    load_nc_subtype_data(startdate, enddate, season)

Retrieve North Carolina subtype counts for the specified season.
"""
function load_nc_subtype_data(startdate::Date, enddate::Date, season::String)
    subtype_path = data_path("interim", "cases", "subtypes_NC_14-25.csv")
    df_subtype = CSV.read(subtype_path, DataFrame, dateformat="yyyy-mm-dd")
    df_subtype = filter(row -> row.season == season, df_subtype)
    df_subtype.date = get_cdc_week_saturday.(df_subtype.date)
    select!(df_subtype, [:date, :flu_A, :flu_B])

    return df_subtype
end

"""
    load_fluview_subtype_data(startdate, enddate)

Return FluView subtype proportions for Region 4 in the given date range.
"""
function load_fluview_subtype_data(startdate::Date, enddate::Date)
    fluview_path = data_path("interim", "cases", "subtypes_FluVIEW-interactive_14-25.csv")
    df_fluview = CSV.read(fluview_path, DataFrame)

    df_fluview = filter(row -> row.REGION == "Region 4", df_fluview)
    df_fluview.date = get_cdc_week_saturday.(df_fluview.YEAR, df_fluview.WEEK)
    df_fluview = filter(row -> startdate ≤ row.date ≤ enddate, df_fluview)

    df_fluview.ratio_H1 = df_fluview."A (H1)" ./ (df_fluview."A (H1)" .+ df_fluview."A (H3)")

    df_ratio = select(df_fluview, [:date, :ratio_H1])
    df_ratio = unique(df_ratio, :date)

    return df_ratio
end

"""
    get_NC_influenza_data(startdate, enddate, season)

Assemble hospitalization, ED, and subtype data aligned by CDC week.
"""
function get_NC_influenza_data(startdate::Date, enddate::Date, season::String)
    df_raw = load_hospitalization_and_ed_data(startdate, enddate)
    df_subtype = load_nc_subtype_data(startdate, enddate, season)
    df_ratio = load_fluview_subtype_data(startdate, enddate)

    df = outerjoin(df_raw, df_subtype, on=:date)
    df[:, [:flu_A, :flu_B]] = coalesce.(df[:, [:flu_A, :flu_B]], 1)

    df.fraction_A = df.flu_A ./ (df.flu_A .+ df.flu_B)
    df.H_inc_A = df.H_inc .* df.fraction_A
    df.H_inc_B = df.H_inc .* (1 .- df.fraction_A)

    dropmissing!(df)

    df_final = select(df, [:date, :H_inc, :I_inc, :H_inc_A, :H_inc_B])
    filter!(row -> startdate ≤ row.date ≤ enddate, df_final)

    df_result = leftjoin(df_final, df_ratio, on=:date)
    df_result.ratio_H1 = coalesce.(df_result.ratio_H1, 1.0)

    df_result.H_inc_AH1 = df_result.H_inc_A .* df_result.ratio_H1
    df_result.H_inc_AH3 = df_result.H_inc_A .* (1 .- df_result.ratio_H1)

    return df_result
end

"""
    concatenate_datasets(seasons, start_calibrations, end_calibrations)

Build per-season incidence matrices and lookup indices for calibration windows.
"""
function concatenate_datasets(seasons, start_calibrations, end_calibrations)
    datasets = Vector{DataFrame}()
    I_dataset = DataFrame(week = collect(1:30))
    H_dataset = DataFrame(week = collect(1:30))
    season2idx = Dict{String, Int}()

    for (start, last, season) in zip(start_calibrations, end_calibrations, seasons)
        start_date = start
        end_date = last - Day(1)
        data = get_NC_influenza_data(start_date, end_date, season)
        push!(datasets, data[:, [:date, :H_inc, :I_inc]])
        data.week = collect(axes(data, 1))
        season2idx[season] = length(datasets)
        H_dataset = innerjoin(H_dataset, rename(data[:, [:week, :H_inc]], :H_inc => season), on = :week)
        I_dataset = innerjoin(I_dataset, rename(data[:, [:week, :I_inc]], :I_inc => season), on = :week)
    end

    return datasets, I_dataset, H_dataset, season2idx
end

function data_pipeline(config::AnalysisConfig)
    start_dates, end_dates = prepare_calibration_window(config.seasons, config.start_month, config.end_month)
    _, I_dataset, H_dataset, season2idx = concatenate_datasets(config.seasons, start_dates, end_dates)
    data = cat(Matrix(select(I_dataset, Not(:week))), Matrix(select(H_dataset, Not(:week))), dims = 3)
    t_span = (0.0, 7.0 * (size(data, 1) - 1))
    population = get_demography(config.population_fips)
    return data, population, t_span, season2idx
end

"""
    select_parameters(season; model="SIR-1S", immunity_linking=false, file)

Pull the optimal parameter row for `season` from the calibration CSV.
"""
function select_parameters(season, model="SIR-1S", immunity_linking=false; file=data_path("interim", "calibration", "single-season-optimal-parameters.csv"))
    pars_0 = CSV.read(file, DataFrame)
    pars_0 = groupby(pars_0, [:model, :immunity_linking])
    pars_model = pars_0[(model, immunity_linking)]
    pars_season = pars_model[:, ["parameter", season]]
    return Dict(row.parameter => row[season] for row in eachrow(pars_season))
end

"""
    convert_seasonal_parameters(parameters, seasons)

Translate per-season parameter tables into a symbol keyed dictionary.
"""
function convert_seasonal_parameters(parameters::DataFrame, seasons::Vector{<:AbstractString})
    param_map = Dict(
        "rho_i" => "ρᵢ",
        "T_h" => "Tₕ",
        "rho_h" => "ρₕ",
        "f_I" => "fᵢ",
        "f_R" => "fᵣ",
        "beta" => "β",
    )

    result = Dict{Symbol, Float64}()
    for (i, season) in enumerate(seasons)
        for row in eachrow(parameters)
            old_key = row.parameter
            value = row[season]
            if occursin("delta_beta_temporal_", old_key)
                idx = parse(Int, split(old_key, "_")[end]) + 1
                result[Symbol("Δβ[$(i), :][$(idx)]")] = value
                result[Symbol("Δβ_raw[$(i), :][$(idx)]")] = 0.5 * (value + 1)
            elseif haskey(param_map, old_key)
                result[Symbol(param_map[old_key])] = value
                result[Symbol("$(param_map[old_key])[$(i)]")] = value
            end
        end
    end
    return result
end

"""
    convert_hyper_parameters(parameters; condition=:exclude_None)

Extract hyper-parameter values from the calibration table into a dictionary.
"""
function convert_hyper_parameters(parameters::DataFrame; condition=:exclude_None)
    param_map = Dict(
        "beta_mu" => "βμ",
        "beta_sigma" => "βσ",
        ["delta_beta_temporal_mu_$(i-1)" => "Δβμ[$(i)]" for i in 1:12]...,
        ["delta_beta_temporal_sigma_$(i-1)" => "Δβσ[$(i)]" for i in 1:12]...,
    )

    result = Dict{Symbol, Float64}()
    for row in eachrow(parameters)
        if haskey(param_map, row.parameter)
            result[Symbol(param_map[row.parameter])] = row[condition]
        end
    end

    return result
end

function get_parameter_names(chain::MCMCChains.Chains)
    return [n for n in names(chain) if !any(isequal(Symbol(n)), [:lp, :logprior, :loglikelihood])]
end

function get_parameter_names(model::Turing.DynamicPPL.Model)
    chain = sample(model, Prior(), 1)
    return get_parameter_names(chain)
end

"""
    get_initial_guess(model, seasons; kwargs...)

Produce an initial parameter assignment compatible with a Turing model.
"""
function get_initial_guess(model, seasons::Vector{<:AbstractString}; model_name="SIR-1S", immunity_linking=false, use_ED_visits=true, identifier="exclude_None")
    parameters = get_parameter_names(model)
    manual_map = Dict(
        [Symbol("Δβ[$i]") => 0.0 for i in 1:12]...,
        [Symbol("raw_Δβ[$i]") => 0.0 for i in 1:12]...,
        [Symbol("α_Δβ[$i]") => 5.0 for i in 1:12]...,
        [Symbol("β_Δβ[$i]") => 5.0 for i in 1:12]...,
    )

    pars_model_0 = CSV.read(data_path("interim", "calibration", "single-season-optimal-parameters.csv"), DataFrame)
    pars_model_0 = filter(row -> row.model == model_name && row.immunity_linking == immunity_linking, pars_model_0)
    seasonal_map = convert_seasonal_parameters(pars_model_0, seasons)

    hyperpars_0 = CSV.read(data_path("interim", "calibration", "hyperparameters.csv"), DataFrame)
    hyperpars_0 = filter(row ->
        row.model == model_name &&
        row.immunity_linking == immunity_linking &&
        row.use_ED_visits == use_ED_visits,
        hyperpars_0)
    hyper_map = convert_hyper_parameters(hyperpars_0; condition=Symbol(identifier))
    param_map = merge(manual_map, seasonal_map, hyper_map)

    return [p => param_map[p] for p in parameters]
end

@inline parameter_root(str::AbstractString) = Symbol(first(split(str, "["; limit=2)))
@inline parameter_root(sym::Symbol) = parameter_root(String(sym))
@inline parameter_indices(str::AbstractString) = [parse(Int, m.match) for m in eachmatch(r"\d+", str)]
@inline parameter_indices(sym::Symbol) = parameter_indices(String(sym))

extract_value(chain::MCMCChains.Chains, name::Symbol, walker::Int) =
    parameter_root(name) => Float64(chain[name][end, walker])

function extract_vector(chain::MCMCChains.Chains, root::Symbol, names::Vector{Symbol}, walker::Int)
    relevant = [n for n in names if length(parameter_indices(n)) == 1]
    isempty(relevant) && return extract_value(chain, root, walker)
    max_idx = maximum(parameter_indices(n)[1] for n in relevant)
    values = zeros(Float64, max_idx)
    for n in relevant
        idx = parameter_indices(n)[1]
        values[idx] = Float64(chain[n][end, walker])
    end
    return root => values
end

function extract_matrix(chain::MCMCChains.Chains, root::Symbol, names::Vector{Symbol}, walker::Int)
    relevant = [n for n in names if length(parameter_indices(n)) == 2]
    isempty(relevant) && return extract_vector(chain, root, names, walker)
    first_dims = [parameter_indices(n)[1] for n in relevant]
    second_dims = [parameter_indices(n)[2] for n in relevant]
    matrix = zeros(Float64, maximum(first_dims), maximum(second_dims))
    for n in relevant
        idx = parameter_indices(n)
        matrix[idx[1], idx[2]] = Float64(chain[n][end, walker])
    end
    return root => matrix
end

function aggregate_parameter_pairs(chain::MCMCChains.Chains, parameter_names::Vector{Symbol}, walker::Int)
    processed = Set{Symbol}()
    pairs = Pair{Symbol, Any}[]

    for name in parameter_names
        root = parameter_root(name)
        root in processed && continue
        push!(processed, root)

        names_for_root = [n for n in parameter_names if parameter_root(n) == root]
        scalar_names = [n for n in names_for_root if isempty(parameter_indices(n))]
        indexed_names = [n for n in names_for_root if !isempty(parameter_indices(n))]

        if isempty(indexed_names)
            push!(pairs, extract_value(chain, scalar_names[1], walker))
            continue
        end

        max_len = maximum(length(parameter_indices(n)) for n in indexed_names)
        if max_len == 1
            push!(pairs, extract_vector(chain, root, indexed_names, walker))
        else
            push!(pairs, extract_matrix(chain, root, indexed_names, walker))
        end
    end

    return pairs
end

function expand_Δβ_matrix(old_matrix::AbstractMatrix, old_αs::AbstractVector, old_βs::AbstractVector, n_Δβ_target::Int)
    prev_n_Δβ = size(old_matrix, 2)
    if prev_n_Δβ == n_Δβ_target
        return copy(old_matrix), copy(old_αs), copy(old_βs)
    else
        @assert n_Δβ_target > prev_n_Δβ
        repeat_factor = max(1, ceil(Int, n_Δβ_target / prev_n_Δβ))
        expanded = zeros(Float64, size(old_matrix, 1), n_Δβ_target)
        for season in axes(old_matrix, 1)
            row = repeat(old_matrix[season, :], inner=repeat_factor)
            expanded[season, :] = row[1:n_Δβ_target]
        end
        expanded_αs = repeat(old_αs, inner=repeat_factor)[1:n_Δβ_target]
        expanded_βs = repeat(old_βs, inner=repeat_factor)[1:n_Δβ_target]
        return expanded, expanded_αs, expanded_βs
    end
end

"""
    build_init_from_prev(prev_chain, n_Δβ_target, n_seasons; noise=0.01)

Recycle the tail of a previous Emcee chain to form initial walkers for the next
stage, expanding the Δβ grid when required.
"""
function build_init_from_prev(prev_chain::MCMCChains.Chains, n_Δβ_target::Int, n_seasons::Int; noise::Real=0.01)
    names = get_parameter_names(prev_chain)
    n_walkers = size(prev_chain, 3)

    init_params = Vector{InitFromParams}(undef, n_walkers)

    for walker in 1:n_walkers
        pairs = aggregate_parameter_pairs(prev_chain, names, walker)

        idx = findfirst(p -> p.first == :Δβ_raw, pairs)
        if isnothing(idx)
            new_Δβ_matrix = fill(0.5, n_seasons, n_Δβ_target) + noise * randn(n_seasons, n_Δβ_target)
            push!(pairs, :Δβ_raw => new_Δβ_matrix)
            push!(pairs, :α_Δβ => fill(50.0, n_Δβ_target) + noise * randn(n_Δβ_target))
            push!(pairs, :β_Δβ => fill(50.0, n_Δβ_target) + noise * randn(n_Δβ_target))
        else
            α_idx = findfirst(p -> p.first == :α_Δβ, pairs)
            β_idx = findfirst(p -> p.first == :β_Δβ, pairs)
            @assert !isnothing(α_idx) && !isnothing(β_idx)
            new_Δβ_matrix, new_α_Δβ, new_β_Δβ = expand_Δβ_matrix(pairs[idx].second, pairs[α_idx].second, pairs[β_idx].second, n_Δβ_target)
            pairs[idx] = :Δβ_raw => new_Δβ_matrix + noise * randn(n_seasons, n_Δβ_target)
            pairs[α_idx] = :α_Δβ => new_α_Δβ + noise * randn(n_Δβ_target)
            pairs[β_idx] = :β_Δβ => new_β_Δβ + noise * randn(n_Δβ_target)
        end

        init_params[walker] = InitFromParams(NamedTuple(pairs), nothing)
    end

    return init_params
end

"""
    run_multistage_emcee(data, population, t_span, config; kwargs...)

Run Emcee over a sequence of Δβ resolutions, reusing posterior samples from one
stage to initialise the next. The `model_builder` callback should accept
`n_Δβ` like `hierarchical_SIR` and `hierarchical_SIR_wo_bounds` do.
"""
function run_multistage_emcee(
    data,
    population,
    t_span,
    config;
    ks = (0, 3, 6, 12),
    steps = (7500, 7500, 7500, 15000),
    noise = 0.005,
    model_builder = hierarchical_SIR_wo_bounds
)
    length(ks) == length(steps) || throw(ArgumentError("ks and steps must have matching lengths"))
    n_seasons = size(data, 2)

    models = Dict(k => model_builder(data, population, t_span; n_Δβ=k) for k in ks)
    init_guess = get_initial_guess(models[ks[end]], config.seasons)
    n_walkers = 2 * length(init_guess)

    chains = Dict{Int, MCMCChains.Chains}()
    prev_chain = nothing

    for (stage_idx, k) in enumerate(ks)
        model = models[k]
        stage_steps = steps[stage_idx]

        if stage_idx == 1
            chains[k] = sample(model, Emcee(n_walkers), stage_steps)
        else
            init_params = build_init_from_prev(prev_chain, k, n_seasons; noise)
            chains[k] = sample(model, Emcee(n_walkers), stage_steps; initial_params=init_params)
        end

        prev_chain = chains[k]
    end

    return chains
end
