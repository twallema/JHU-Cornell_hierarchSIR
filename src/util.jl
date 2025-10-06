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

"""
    get_initial_guess(model, seasons; kwargs...)

Produce an initial parameter assignment compatible with a Turing model.
"""
function get_initial_guess(model, seasons::Vector{<:AbstractString}; model_name="SIR-1S", immunity_linking=false, use_ED_visits=true, identifier="exclude_None")
    chain = sample(model, Prior(), 1)
    parameters = chain.name_map.parameters
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
