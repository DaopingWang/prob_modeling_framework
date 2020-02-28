include("custom_distributions/custom_distribution_lib.jl")
include("deprecated/Util.jl")

using SQLite
using Gen
using CSV
using DataFrames: DataFrame, Missing
using LinearAlgebra
using Statistics
using Logging
using Random

function parse_data(db)
    local realization_df
    local states
    local cities
    local zips
    local associated_cities

    realization_df = SQLite.Query(db, "SELECT dirty_state, dirty_city, dirty_zip, rent FROM rent_table") |> DataFrame

    states = SQLite.Query(db, "SELECT DISTINCT dirty_state FROM rent_table") |> DataFrame
    states.occurrence = [sum(realization_df.dirty_state .== st) for st in states.dirty_state]
    states.associated_cities = [realization_df[realization_df.dirty_state .== st, :dirty_city] for st in states.dirty_state]

    cities = SQLite.Query(db, "SELECT DISTINCT dirty_city FROM rent_table") |> DataFrame
    cities.occurrence = [sum(realization_df.dirty_city .== c) for c in cities.dirty_city]
    cities.associated_states = [realization_df[realization_df.dirty_city .== c, :dirty_state] for c in cities.dirty_city]
    cities.associated_zips = [realization_df[realization_df.dirty_city .== c, :dirty_zip] for c in cities.dirty_city]

    zips = SQLite.Query(db, "SELECT DISTINCT dirty_zip FROM rent_table") |> DataFrame
    zips.occurrence = [sum(realization_df.dirty_zip .== z) for z in zips.dirty_zip]
    zips.associated_cities = [realization_df[realization_df.dirty_zip .== z, :dirty_city] for z in zips.dirty_zip]

    (realization_df, states, cities, zips)
end

# BROKEN
function softmax(probs)
    denominator = sum(map(exp, probs))
    map(x -> exp(x)/denominator, probs)
end

@gen function statistics(df, attr)
    @assert typeof(df) == DataFrame
    @assert typeof(attr) == Symbol

    attr_dup_rows = df[:, attr]
    attr_unique_rows = unique(attr_dup_rows)
    attr_oc = [sum(attr_dup_rows .== val) for val in attr_unique_rows]
    total_attr_oc = sum(attr_oc)

    probs = ones(length(attr_unique_rows))
    for (i, val) in enumerate(attr_unique_rows)
        prob = beta(attr_oc[i], total_attr_oc - attr_oc[i])
        probs[i] *= prob
    end
    realization = @trace(named_categorical(normalize(probs, 1), map(Symbol,attr_unique_rows)), :realization)
end

@gen function functional_dependency(df, attr_1, attr_2, val_1, care_global_oc)
    @assert typeof(df) == DataFrame
    @assert typeof(attr_1) == typeof(attr_2) == Symbol
    @assert typeof(care_global_oc) == Bool
    # all observed values, with duplicates
    attr_1_dup_rows = df[:, attr_1]
    attr_2_dup_rows = df[:, attr_2]
    # unique attr_2 values
    attr_2_unique_rows = unique(attr_2_dup_rows)
    # all observed attr_2 values that co-occurred with val_1, with duplicates
    associated_attr_2 = df[attr_1_dup_rows .== val_1, attr_2]

    if care_global_oc
        global_attr_2_oc = [sum(attr_2_dup_rows .== val) for val in attr_2_unique_rows]
        total_global_attr_2_oc = sum(global_attr_2_oc)
    end
    local_attr_2_oc = [if val in associated_attr_2
                          sum(associated_attr_2 .== val)
                       else
                          1
                       end for val in attr_2_unique_rows]
    total_local_attr_2_oc = sum(local_attr_2_oc)

    attr_2_probs = ones(length(attr_2_unique_rows))
    for (i, val) in enumerate(attr_2_unique_rows)
        local_val_prob = beta(local_attr_2_oc[i], total_local_attr_2_oc - local_attr_2_oc[i])
        attr_2_probs[i] *= local_val_prob
        if care_global_oc
            global_val_prob = beta(global_attr_2_oc[i], total_global_attr_2_oc - global_attr_2_oc[i])
            attr_2_probs[i] *= global_val_prob
        end
    end

    #attr_2_probs = softmax(local_attr_2_oc)
    realization = @trace(named_categorical(normalize(attr_2_probs, 1), map(Symbol, attr_2_unique_rows)), :realization)
end

# Special FD-Case of multiple lefthand side arguments
# e.g. A=a, B=b, C=c -> Y=y
@gen function co_occurrence(df, attr_left, attr_right, val_left, care_global_oc)
    @assert typeof(attr_left) == Array{Symbol, 1}
    @assert typeof(attr_right) == Symbol
    @assert typeof(care_global_oc) == Bool

    attr_left_dup_rows = [collect(skipmissing(df[:, attr])) for attr in attr_left]
    attr_right_dup_rows = collect(skipmissing(df[:, attr_right]))
    attr_right_unique_rows = unique(attr_right_dup_rows)
    # all observed attr_2 values that co-occurred with val_1, with duplicates
    mask = [true for _=1:size(df, 1)]
    for (i, val) in enumerate(attr_left)
        mask = mask .& (attr_left_dup_rows[i] .== val_left[i])
    end
    associated_attr_right = df[mask, attr_right]
    if care_global_oc
        global_attr_right_oc = [sum(attr_right_dup_rows .== val) for val in attr_right_unique_rows]
        total_global_attr_right_oc = sum(global_attr_right_oc)
    end
    # TODO We have a lot of data in contrast to the low number of occurrence of each attr. val
    # Think about how to reward occurred value and how to punish others without
    # stating an hard constraint
    local_attr_right_oc = [if val in associated_attr_right
                              sum(associated_attr_right .== val) * 100
                           else
                              1
                           end for val in attr_right_unique_rows]
    total_local_attr_right_oc = sum(local_attr_right_oc)
    attr_right_probs = ones(length(attr_right_unique_rows))
    for (i, val) in enumerate(attr_right_unique_rows)
        local_val_prob = beta(local_attr_right_oc[i], total_local_attr_right_oc - local_attr_right_oc[i])
        attr_right_probs[i] *= local_val_prob
        if care_global_oc
            global_val_prob = beta(global_attr_right_oc[i], total_global_attr_right_oc - global_attr_right_oc[i])
            attr_right_probs[i] *= global_val_prob
        end
    end
    realization = @trace(named_categorical(normalize(attr_right_probs, 1), map(Symbol, attr_right_unique_rows)), :realization)
end

# Given lefthand side attribute values, sample numerical value for one righthand
# side attribute
@gen function numerical_functional_dependency(df, attr_left, attr_right, val_left, care_global_oc)
    @assert typeof(attr_left) == Array{Symbol, 1}
    @assert typeof(attr_right) == Symbol
    @assert typeof(care_global_oc) == Bool

    # all lefthand side attribute values w/o missings
    attr_left_dup_rows = [collect(skipmissing(df[:, attr])) for attr in attr_left]

    # all observed attr_2 values that co-occurred with val_1, with duplicates
    mask = [true for _=1:size(df, 1)]
    for (i, val) in enumerate(attr_left)
        mask = mask .& (attr_left_dup_rows[i] .== val_left[i])
    end
    associated_attr_right = df[mask, attr_right]

    # mean and std of righthand side values that occurred with given left values
    local_attr_right_mean = mean(collect(skipmissing(associated_attr_right)))
    local_attr_right_std = std(collect(skipmissing(associated_attr_right)))
    local_attr_right_std = isnan(local_attr_right_std) ? 0.01 : local_attr_right_std

    if care_global_oc || isnan(local_attr_right_mean)
        global_attr_right_mean = mean(collect(skipmissing(df[:, attr_right])))
        global_attr_right_std = std(collect(skipmissing(df[:, attr_right])))
        global_attr_right_std = isnan(global_attr_right_std) ? 0.01 : global_attr_right_std

        mean_var = mean([local_attr_right_mean, global_attr_right_mean])
        #std_var = mean([local_attr_right_std, global_attr_right_std])
        std_var = global_attr_right_std
        return @trace(normal(mean_var, std_var), :realization)
    end
    realization = @trace(normal(local_attr_right_mean, local_attr_right_std), :realization)
end

#@gen function rent_model_advance(realization_df, states, cities, zips)
@gen function rent_model_advance(realization_df)
    @info "-----------------ADV"
    realized_state = @trace(statistics(realization_df,
                                       :dirty_state), :state)
    @info "State done: $realized_state"

    #= realized_city = @trace(functional_dependency(realization_df,
                                                 :dirty_state,
                                                 :dirty_city,
                                                 String(realized_state),
                                                 true), :realized_city)
    @info "City done: $realized_city"=#

    co_oc_city = @trace(co_occurrence(realization_df,
                                     [:dirty_state,],
                                     :dirty_city,
                                     [String(realized_state)],
                                     true), :city)
    @info "COOC City: $co_oc_city"

    #= realized_zip = @trace(functional_dependency(realization_df,
                                                :dirty_city,
                                                :dirty_zip,
                                                String(realized_city),
                                                true), :realized_zip)
    @info "Zip done: $realized_zip"=#

    co_oc_zip = @trace(co_occurrence(realization_df,
                                    [:dirty_state, :dirty_city],
                                    :dirty_zip,
                                    [String(realized_state), String(co_oc_city)],
                                    true), :zip)
    @info "COOC Zip: $co_oc_zip"

    realized_rent = @trace(numerical_functional_dependency(realization_df,
                                                        [:dirty_city, :dirty_zip],
                                                        :rent,
                                                        [String(co_oc_city), String(co_oc_zip)],
                                                        true), :rent)
    @info "COOC Rent: $realized_rent"
end

# Supervised method
@gen function rent_model(realization_df, states, cities, zips)
    # For modeling the pdf of state and city vars, we use categorical with thetas
    # drawn from beta dist. depending on total number of occurrence. Also incorperate
    # !(t1.city = t2.city -> t1.state != t2.state) softly
    @info "--------------------"
    # CHOOSE STATE - global statistics
    # - Occurrence statistics (global)
    # - Each state has a probability drawn from a beta dist. with alpha=occurrence
    # and beta=total number of data entries - occurrence
    global_state_oc = states.occurrence
    state_probs = ones(length(states.dirty_state))
    for (i, st) in enumerate(states.dirty_state)
        # what occurred stays relevant
        prob = @trace(beta(global_state_oc[i], sum(global_state_oc) - global_state_oc[i]), :state_data => i => :prob)
        state_probs[i] *= prob
    end

    state_names = map(Symbol, states.dirty_state)
    realized_state = @trace(named_categorical(normalize(state_probs, 1), state_names), :realized_state)
    @info "State done: $realized_state"

    # ERROR DETECTION AND CLEANING - NOT EASY!!!
    # - For each attribute value, we want to know if it is clean.
    # - The probability of a value to be clean is initially proportional to number of occurrence
    # - I.e., if we assume 10% of all values are dirty, those values with num of oc
    # in the lowest 10% have great chance of being dirty
    #for (i, clean_prob) in enumerate(state_probs)
        # TODO implement this
        # for each dirty_city, draw is_clean from bernoulli(beta) (?)
        # if realized state is dirty, then draw :intended_state from probs but without the dirty entries
    #end

    # CHOOSE CITY - FD, FD statistics, global statistics
    # - FD with ZIP, occurrence statistics (global and local-FD)
    # - First, pick possible cities given state with soft constraint
    # TODO think about turning score params into trainable params

    # number of occurrence of each city in db

    global_city_oc = cities.occurrence
    # cities that have co-occurred with chosen realized state
    associated_cities = states[states.dirty_state .== String(realized_state), :].associated_cities[1]
    # number of co-occurrence of each city with realized state
    local_city_oc = [if c in associated_cities
                        sum(associated_cities .== c)
                     else
                        0.1
                     end for c in cities.dirty_city]
    # preallocate probability list
    city_probs = ones(length(cities.dirty_city))
    # use beta(number_of_occurrence, total_occurrence - number_of_occurrence)
    # for probability of each observed city, globally and locally
    for (i, c) in enumerate(cities.dirty_city)
        global_city_prob = @trace(beta(global_city_oc[i], sum(global_city_oc) - global_city_oc[i]), :city_data => i => :global_prob)
        local_city_prob = @trace(beta(local_city_oc[i], sum(local_city_oc) - local_city_oc[i]), :city_data => i => :local_prob)
        city_probs[i] *= global_city_prob * local_city_prob
    end
    city_names = map(Symbol, cities.dirty_city)
    realized_city = @trace(named_categorical(normalize(city_probs, 1), city_names), :realized_city)
    @info "City done: $realized_city"

    # CHOOSE ZIP - FD, FD statistics, global statistics
    # Functional dependency with CITY, occurrence statistics (global and local-FD)
    # global number of occurrence
    global_zip_oc = zips.occurrence
    # local number of occurrence
    # local means under the constraint of given realized city
    associated_zips = cities[cities.dirty_city .== String(realized_city), :].associated_zips[1]
    local_zip_oc = [if z in associated_zips
                        sum(associated_zips .== z)
                    else
                        0.1
                    end for z in zips.dirty_zip]
    # combine global and local probability (global -> prior, local -> likelihood)
    zip_probs = ones(length(zips.dirty_zip))
    for (i, z) in enumerate(zips.dirty_zip)
        global_zip_prob = @trace(beta(global_zip_oc[i], sum(global_zip_oc) - global_zip_oc[i]), :zip_data => i => :global_prob)
        loc_zip_prob = @trace(beta(local_zip_oc[i], sum(local_zip_oc) - local_zip_oc[i]), :zip_data => i => :loc_prob)
        zip_probs[i] *= loc_zip_prob * global_zip_prob
    end
    zip_names = map(Symbol, zips.dirty_zip)
    realized_zip = @trace(named_categorical(normalize(zip_probs, 1), zip_names), :realized_zip)
    @info "Zip_done: $realized_zip"

    # SAMPLE RENT - co-occurrence statistics
    # we could choose params for realized_rent = normal(mean, std) from 3 places:
    # 1. mean and std of rents from realized city
    realized_city_rents = realization_df[realization_df.dirty_city .== String(realized_city), :].rent
    # 2. mean and std of rents from inteded zip
    realized_zip_rents = realization_df[realization_df.dirty_zip .== String(realized_zip), :].rent
    # 3. mean and std of rents from realized zip-city co-occurrences
    co_oc_rents = realization_df[(realization_df.dirty_city .== String(realized_city)) .& (realization_df.dirty_zip .== String(realized_zip)), :].rent

    # list of occurrence of realized city and realized zip (more frequent -> more credible)
    oc = [cities.occurrence[cities.dirty_city .== String(realized_city)][1], zips.occurrence[zips.dirty_zip .== String(realized_zip)][1]]
    mean_rents = map(mean, [realized_city_rents, realized_zip_rents])
    std_rents = map(std, [realized_city_rents, realized_zip_rents])
    if (n_co_oc = length(co_oc_rents)) > 0
        # the chance of choosing mean and std from realized city/zip is 0.5 * normalized number of occurence
        # the chance of choosing mean and std from co-occurred intention is 0.5
        push!(oc, sum(oc))
        push!(mean_rents, mean(co_oc_rents))
        std_co_oc_rents = std(co_oc_rents)
        push!(std_rents, std_co_oc_rents)
    end
    std_rents = [if isnan(i) 0.5 else i end for i in std_rents]
    mean_rents = map(Symbol, mean_rents)
    mean_par = @trace(named_categorical(normalize(oc, 1), mean_rents), :rent_mean_par)
    std_par = std_rents[mean_rents .== mean_par][1]
    realized_rent = @trace(normal(parse(Float64, String(mean_par)), std_par), :realized_rent)
    @info "Rent done: $realized_rent"
end


db = SQLite.DB()
rent_table = CSV.File(joinpath(@__DIR__, "../data/rent_data.csv")) |> SQLite.load!(db, "rent_table")
(realization_df, states, cities, zips) = parse_data(db)

# Simulation
#traces = [Gen.simulate(rent_model_advance, (realization_df, states, cities, zips)) for _=1:2]
#for trace in traces
    # println("\n$(trace[:realized_state]) \n$(trace[:realized_city]) \n$(trace[:realized_zip]) \n$(trace[:realized_rent])")
#end

constraints = Gen.choicemap()
#constraints[:state => :realization] = Symbol("vBayern")
constraints[:city => :realization] = Symbol("Muenchen")
constraints[:zip => :realization] = Symbol("80331")
constraints[:rent => :realization] = 11.73

dirty_constraints = Gen.choicemap()
dirty_constraints[:state => :realization] = Symbol("Bayern")
dirty_constraints[:city => :realization] = Symbol("Muenchen")
dirty_constraints[:zip => :realization] = Symbol("80331")
dirty_constraints[:rent => :realization] = 11.73
#=
for _=1:3
    println("--------------------")
    (trace, weight) = Gen.generate(rent_model_advance, (realization_df, states, cities, zips), constraints)
    println("$(trace[:state => :realization])
            $(trace[:city => :realization])
            $(trace[:zip => :realization])
            $(trace[:rent => :realization])")
    println("Log-likelihood: $weight\n")
    (trace, weight) = Gen.generate(rent_model_advance, (realization_df, states, cities, zips), dirty_constraints)
    println("DIRTY: $(trace[:state => :realization])
                    $(trace[:city => :realization])
                    $(trace[:zip => :realization])
                    $(trace[:rent => :realization])")
    println("Log-likelihood: $weight\n")
end=#

function do_inference(model, args, constraints, num_iter)
    (trace, lml_est) = Gen.importance_resampling(model, args, constraints, num_iter)
    return (trace, lml_est)
end

#trace, lml_est = do_inference(rent_model_advance, (realization_df, states, cities, zips), constraints, 50)
#println("$(trace[:state => :realization]) \n$(trace[:city => :realization]) \n$(trace[:zip => :realization]) \n$(trace[:rent => :realization])")
#println("Log-likelihood: $lml_est")

#disable_logging(LogLevel(10))

function k_most_improbable(k, observation_df, attrs, model)
    @assert typeof(attrs) == Dict{Symbol, Symbol}
    @assert typeof(observation_df) == DataFrame

    k_trace_list = []
    k_score_list = []
    for i in 1:size(observation_df, 1)
        constraints = Gen.choicemap()
        for key in keys(attrs)
            if key == :rent || key == :ek_amount
                constraints[key => :realization] = observation_df[i, attrs[key]]
            else
                constraints[key => :realization] = Symbol(observation_df[i, attrs[key]])
            end
        end

        (trace, weight) = Gen.generate(model, (observation_df, ), constraints)
        if length(k_score_list) < k
            push!(k_score_list, weight)
            push!(k_trace_list, trace)
            continue
        end
        max_score = maximum(k_score_list)
        if weight < max_score
            max_i = findall(k_score_list .== max_score)[1]
            deleteat!(k_score_list, max_i)
            deleteat!(k_trace_list, max_i)
            push!(k_score_list, weight)
            push!(k_trace_list, trace)
        end
    end
    return (k_trace_list, k_score_list)
end

#=
attrs = Dict(:state => :dirty_state,
            :city => :dirty_city,
            :zip => :dirty_zip,
            :rent => :rent)
(k_traces, k_scores) = k_most_improbable(10, realization_df, attrs, rent_model_advance)

for trace in k_traces
    println("--------------------")
    println("$(trace[:state => :realization])\n$(trace[:city => :realization])\n$(trace[:zip => :realization])\n$(trace[:rent => :realization])")
    score = get_score(trace)
    println("Log-likelihood: $score\n")
end

b_constraints = Gen.choicemap()
b_constraints[:state => :realization] = :Bayrn
b_constraints[:city => :realization] = :Baxreuth
b_constraints[:zip => :realization] = Symbol(95444)
b_constraints[:rent => :realization] = 8.089619123554883
(trace, weight) = generate(rent_model_advance, (realization_df, ), b_constraints)
score = get_score(trace)
println(score)
=#
