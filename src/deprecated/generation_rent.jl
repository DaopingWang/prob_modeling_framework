cd(dirname(@__FILE__))
include("custom_distributions/custom_distribution_lib.jl")
include("deprecated/Util.jl")
include("modeling/util.jl")

using SQLite
using Gen
using StaticArrays
using CSV
using DataFrames: DataFrame
using LinearAlgebra

@gen function rent_generation_model()
    HEATING_COST_DICT = Dict("Natural" => 10.0,
                        "Oil" => 12.0,
                        "District" => 13.0,
                        "None" => 3.0)

    STATE_MEAN_CONSTRUCT_YEAR_DICT = Dict("Bayern" => 1989,
                                        "Nordrhein-Westfalen" => 1959,
                                        "Baden-Wuerttemberg" => 1959,
                                        "Niedersachsen" => 1949,
                                        "Hessen" => 1939)

    state_df = SQLite.Query(db, "SELECT DISTINCT state FROM rent_table") |> DataFrame
    num_states = size(state_df, 1)

    # States and cities are uniformly sampled from
    i = uniform_discrete(1, num_states)
    intended_state = state_df[i, 1]

    city_df = SQLite.Query(db, "SELECT
    city,
    zip,
    zip_start,
    zip_end,
    mean_rent,
    no_heating
    FROM rent_table WHERE state='$intended_state'") |> DataFrame
    num_cities = size(city_df, 1)

    i = uniform_discrete(1, num_cities)
    intended_city = city_df.city[i]
    #intended_zip = city_df[i, 2]
    zip_start = city_df.zip_start[i]
    zip_end = city_df.zip_end[i]
    intended_zip = uniform_discrete(zip_start, zip_end)
    mean_rent_truth = city_df.mean_rent[i] + 2.5# - 2.
    no_heating = city_df.no_heating[i]

    # Construction year has influence on base rent
    # Buildings built short after WW2 are crap
    #construct_year = uniform_discrete(1920, 2010)
    mean_rent = mean_rent_truth
    construct_year = normal(STATE_MEAN_CONSTRUCT_YEAR_DICT[intended_state], 20)
    construct_year = convert(Int, floor(construct_year))
    if construct_year < 1949
        mean_rent += 1.5
    elseif construct_year < 1969
        mean_rent -= 1.
    elseif construct_year > 1990
        mean_rent += (construct_year-1990) * 0.25
    end

    # The smaller the appartement, the more expensive the unit rent gets
    living_space = uniform_continuous(20, 200)
    if living_space < 100
        mean_rent += (100 - living_space) * 0.0425
    else
        mean_rent -= (living_space - 100) * 0.0425
    end

    base_rent = living_space * mean_rent

    additional_cost = 0
    heating_cost = 0

    # Poor cities or countryside could have houses w/o heating
    # New buildings always have heating
    # Different heating types have different unit price
    heating_type = nothing
    if (construct_year < 1989) && bernoulli(no_heating)
        heating_type = "None"
    else
        i = uniform_discrete(1, 3)
        if i == 1
            heating_type = "Natural"
        elseif i == 2
            heating_type = "Oil"
        else
            heating_type = "District"
        end
    end
    heating_cost = HEATING_COST_DICT[heating_type] * living_space

    # Balcony influences the price a bit
    has_balcony = false
    if bernoulli(0.1)
        has_balcony = true
        additional_cost += normal(50.0, 10.0)
    end

    # Older buildings are unprobable to have parking possibilities
    has_parking = false
    parking_prob = construct_year > 1995 ? 0.75 : 0.3
    if bernoulli(parking_prob)
        has_parking = true
        additional_cost += normal(75.0, 5.0)
    end

    total_rent = base_rent# + additional_cost + heating_cost
    total_rent = normal(total_rent, 20.)

    return (intended_state,
            intended_city,
            intended_zip,
            round(living_space; digits=2),
            construct_year,
            #heating_type,
            #has_parking,
            #has_balcony,
            #round(mean_rent_truth; digits=2),
            round(total_rent; digits=2))
end

@gen function simple_rent(db)
    state_df = SQLite.Query(db, "SELECT DISTINCT state FROM rent_table") |> DataFrame
    num_states = size(state_df, 1)

    # States and cities are uniformly sampled from
    i = uniform_discrete(1, num_states)
    state = state_df[i, 1]

    city_df = SQLite.Query(db, "
    SELECT
        city, zip_start, zip_end, mean_rent
    FROM
        rent_table
    WHERE
        state='$state'") |> DataFrame

    num_cities = size(city_df, 1)
    i = uniform_discrete(1, num_cities)
    city = city_df.city[i]

    zip_start = city_df.zip_start[i]
    zip_end = city_df.zip_end[i]
    zip = uniform_discrete(zip_start, zip_end)

    rent = normal(city_df.mean_rent[i], 1.)
    return (state, city, zip, round(rent; digits=2))
end

db = SQLite.DB()
rent_table = CSV.File(joinpath(@__DIR__, "../data/cities.csv")) |> SQLite.load!(db, "rent_table")

#######################################################
# Simulation
# enriched rent model
#=rent_data = [rent_generation_model() for _=1:100000]
df = DataFrame(state = String[],
                city = String[],
                zip = Int[],
                living_space = Float32[],
                construct_year = Int[],
                #heating_type = String[],
                #has_parking = Bool[],
                #has_balcony = Bool[],
                #mean_rent_truth = Float32[],
                rent = Float32[])=#

#######################################################
# neo rent model
neo_data = [rent_generation_model() for _=1:200000]
neo_df = DataFrame(state = String[],
                    city = String[],
                    zip = Int[],
                    living_space = Float32[],
                    construct_year = Int[],
                    rent = Float32[])
for e in neo_data
    push!(neo_df, e)
end

io_mask = [false for _=1:size(neo_df, 1)]
io_dict = Dict()
for (i, c) in enumerate(neo_df.city)
    key_list = keys(io_dict)
    io_dict[c] = c in key_list ? (io_dict[c] + 1) : 1
    # sparsity control && rent positive && living_Space interval
    if (io_dict[c] < 31) && (neo_df.rent[i] > 100) && (abs(neo_df.living_space[i] - 100) < 100)
        io_mask[i] = true
    end
end
small_neo_df = neo_df[io_mask, :]

#######################################################
function eval_negative_sampler(df::DataFrame, pivot::Symbol, attrs::Array, n::Int, p)
    attrs_values_unique = [unique(df[:, attr]) for attr in attrs]
    n_s_df_rows = []
    sampled = false

    for i = 1:n
        entry_i = uniform_discrete(1, size(df, 1))
        n_s_buffer = deepcopy(df[entry_i, :])
        for (j, attr) in enumerate(attrs)
            if (!sampled)&&(j==length(attrs))
                # no neg samp yet, last attr reached
                sampled = true
            elseif bernoulli(p)
                sampled = true
            else
                # dont neg samp this attr
                continue
            end
            pos_attr_vals = unique(df[df[:, pivot].==n_s_buffer[pivot], attr])
            neg_attr_vals = filter(x->x ∉ pos_attr_vals, attrs_values_unique[j])
            n_s_buffer[attr] = uniform_categorical(neg_attr_vals)
            if attr == :rent
                #println("get")
                n_s_buffer[attr] += 2000
                #println(n_s_buffer[attr])
            end
        end
        push!(n_s_df_rows, deepcopy(n_s_buffer))
    end
    return n_s_df_rows
end

function generate_eval_rent_data(sampled_df, sparsity, k, p)
    io_mask = [false for _=1:size(neo_df, 1)]
    io_dict = Dict()
    for (i, c) in enumerate(sampled_df.city)
        key_list = keys(io_dict)
        io_dict[c] = c in key_list ? (io_dict[c] + 1) : 1
        # sparsity control && rent positive && living_Space interval
        if (io_dict[c] < sparsity+1) && (sampled_df.rent[i] > 100) && (abs(sampled_df.living_space[i] - 100) < 100)
            io_mask[i] = true
        end
    end
    small_neo_df = sampled_df[io_mask, :]

    orig_attrs = names(small_neo_df)
    small_neo_df.negative_sample = [0 for _=1:size(small_neo_df, 1)]
    kk = convert(Int, ceil(k*size(small_neo_df, 1)))
    #n_s_rows = negative_sampler(small_neo_df[:, orig_attrs], :city, [:living_space, :rent], 5)
    n_s_rows = eval_negative_sampler(small_neo_df[:, orig_attrs], :city, [:zip, :state], kk, p)
    n_s_df = DataFrame()
    for row in n_s_rows
        push!(n_s_df, row)
    end
    n_s_df.negative_sample = [1 for _=1:size(n_s_df, 1)]
    small_neo_df = vcat(small_neo_df, n_s_df)
    return small_neo_df
end

ss = [10, 50, 100, 200, 300]
kk = [0.01, 0.1, 0.2, 0.3, 0.4]
pp = [0.2, 0.5, 0.8, 1.0]

for s in ss
    eval_df = generate_eval_rent_data(neo_df, s, 0.1, 0.2)
    file_name = "sparsity_"*string(s)*"_k_"*string(01)*"_p_"*string(02)*".csv"
    CSV.write(file_name, eval_df)
end

for k in kk
    eval_df = generate_eval_rent_data(neo_df, 50, k, 0.2)
    file_name = "sparsity_"*string(50)*"_k_"*string(k)*"_p_"*string(02)*".csv"
    CSV.write(file_name, eval_df)
end

for p in pp
    eval_df = generate_eval_rent_data(neo_df, 50, 0.1, p)
    file_name = "sparsity_"*string(50)*"_k_"*string(01)*"_p_"*string(p)*".csv"
    CSV.write(file_name, eval_df)
end

function generate_eval_ek_data(sampled_df, sparsity, k, p)
    small_neo_df = deepcopy(sampled_df)

    orig_attrs = names(small_neo_df)
    small_neo_df.negative_sample = [0 for _=1:size(small_neo_df, 1)]
    kk = convert(Int, ceil(k*size(small_neo_df, 1)))
    #n_s_rows = negative_sampler(small_neo_df[:, orig_attrs], :city, [:living_space, :rent], 5)
    n_s_rows = eval_negative_sampler(small_neo_df[:, orig_attrs], :keywords, [:set_id, :ek_amount, :manufacturer_name], kk, p)
    n_s_df = DataFrame()
    for row in n_s_rows
        push!(n_s_df, row)
    end
    n_s_df.negative_sample = [1 for _=1:size(n_s_df, 1)]
    small_neo_df = vcat(small_neo_df, n_s_df)
    return small_neo_df
end

for k in kk
    eval_df = generate_eval_ek_data(good_df, 30, k, 0.2)
    file_name = "sparsity_"*string(30)*"_k_"*string(k)*"_p_"*string(0.2)*".csv"
    CSV.write(file_name, eval_df)
end

for p in pp
    eval_df = generate_eval_ek_data(good_df, 30, 0.1, p)
    file_name = "sparsity_"*string(30)*"_k_"*string(0.05)*"_p_"*string(p)*".csv"
    CSV.write(file_name, eval_df)
end

#######################################################
# neg sampling
orig_attrs = names(small_neo_df)
small_neo_df.negative_sample = [0 for _=1:size(small_neo_df, 1)]
#n_s_rows = negative_sampler(small_neo_df[:, orig_attrs], :city, [:living_space, :rent], 5)
kk = 0.9
n_s_rows = neo_negative_sampler(small_neo_df[:, orig_attrs], :city, [:zip, :rent, :state], convert(Int, ceil(kk*size(small_neo_df, 1))))
n_s_df = DataFrame()
for row in n_s_rows
    push!(n_s_df, row)
end
n_s_df.negative_sample = [1 for _=1:size(n_s_df, 1)]
small_neo_df = vcat(small_neo_df, n_s_df)
#CSV.write("NEGSAMP_neo_rent_60_per_city.csv", small_neo_df)

#######################################################
# simple rent model
#=
rent_data = [simple_rent(db) for _=1:10000]
df = DataFrame(state = String[],
                city = String[],
                zip = Int[],
                rent = Float32[])
for e in rent_data
    push!(df, e)
end

#CSV.write("enriched_rent_data_50000.csv", df)

io_mask = [false for _=1:size(df, 1)]
io_dict = Dict()
for (i, c) in enumerate(df.city)
    key_list = keys(io_dict)
    io_dict[c] = c in key_list ? (io_dict[c] + 1) : 1
    if (io_dict[c] < 20) #&& (df.rent[i] > 100) && (abs(df.living_space[i] - 100) < 100)
        io_mask[i] = true
    end
end
small_df = df[io_mask, :]

# neg sampling
orig_attrs = names(small_df)
small_df.negative_sample = [0 for _=1:size(small_df, 1)]
n_s_rows = negative_sampler(small_df[:, orig_attrs], :city, [:rent], 2)
n_s_df = DataFrame()
for row in n_s_rows
    push!(n_s_df, row)
end
n_s_df.negative_sample = [1 for _=1:size(n_s_df, 1)]
small_df = vcat(small_df, n_s_df)
CSV.write("NEGSAMP_simple_rent_20_per_city.csv", small_df)
=#
#################################

@gen function rent_model()
    # CHOOSE STATE
    states = SQLite.Query(db, "SELECT DISTINCT state, state_population FROM rent_table") |> DataFrame
    state_names = map(Symbol,states.state)
    state_pop = states.state_population
    intended_state = @trace(named_categorical(LinearAlgebra.normalize(state_pop, 1), state_names), :intended_state)

    # CHOOSE CITY AND ZIP PROPORTIONALLY TO POPULATION AND NUMBER OF TYPOS
    cities = SQLite.Query(db, "SELECT city, population, zip FROM rent_table WHERE state='$intended_state'") |> DataFrame
    city_names = map(Symbol, cities.city)
    city_pop = cities.population
    intended_city = @trace(named_categorical(LinearAlgebra.normalize(city_pop, 1), city_names), :intended_city)

    zip_names = map(Symbol, cities.zip)
    zip_prob = [if city!=String(intended_city) 0. else 1. end for city in cities.city]
    intended_zip = @trace(named_categorical(zip_prob, zip_names), :intended_zip)
    #city_zip = (SQLite.Query(db, "SELECT zip FROM rent_table WHERE city='$intended_city'") |> DataFrame)[1, 1]
    #intended_zip = @trace(degenerate_distribution(Symbol(city_zip)), :intended_zip)

    # SAMPLE RENT
    mean_rent = (SQLite.Query(db, "SELECT mean_rent FROM rent_table WHERE city='$intended_city'") |> DataFrame)[1, 1]
    rent = @trace(Gen.normal(mean_rent, 1.), :rent)

    # ADD TYPOS
    dirty_city = @trace(edit_error_distribution(intended_city), :dirty_city)
    dirty_state = @trace(edit_error_distribution(intended_state), :dirty_state)
    dirty_zip = @trace(edit_error_distribution(Symbol(intended_zip)), :dirty_zip)
end

#=
db = SQLite.DB()
rent_table = CSV.File(joinpath(@__DIR__, "../data/cities.csv")) |> SQLite.load!(db, "rent_table")

# Simulation
traces = [Gen.simulate(rent_model, ()) for _=1:1000]
#for trace in traces
    #println("\n$(trace[:intended_state]) \n$(trace[:intended_city]) \n$(trace[:intended_zip]) \n$(trace[:rent])")
#end


# Write generated data
df = DataFrame(intended_state = String[],
                #dirty_state = String[],
                intended_city = String[],
                #dirty_city = String[],
                intended_zip = String[],
                #dirty_zip = String[],
                rent = String[])
for trace in traces
    tuple = map(String, [#trace[:dirty_state],
                        trace[:intended_state],
                        #trace[:dirty_city],
                        trace[:intended_city],
                        #trace[:dirty_zip],
                        trace[:intended_zip],
                        Symbol(trace[:rent])])
    push!(df, tuple)
end

CSV.write("rent_data_1000.csv", df)

=#
# Inference
constraints = Gen.choicemap()

# Cleaning
#constraints[:dirty_state] = Symbol("Bayrn") # Hessen
#constraints[:dirty_city] = Symbol("offenbach am main")
#constraints[:dirty_zip] = Symbol("60331")
#constraints[:rent] = 13.16    # rent = 4EUR/m2

# Error detection
constraints[:intended_state] = Symbol("Bayern")
constraints[:intended_city] = Symbol("Hof")
#constraints[:dirty_zip] = Symbol(60596)
#constraints[:rent] = 14.

# generate one sample, evaluate its probability
#for _=1:10
#    (trace, weight) = Gen.generate(rent_model, (), constraints)
#    println("$(trace[:intended_state]) \n$(trace[:intended_city]) \n$(trace[:intended_zip]) \n$(trace[:rent])")
#    println("Log-likelihood: $weight")
#end

# INFERENCE
function do_inference(model, args, constraints, num_iter)
    (trace, lml_est) = Gen.importance_resampling(rent_model, args, constraints, num_iter)
    return (trace, lml_est)
end

#trace, lml_est = do_inference(rent_model, (), constraints, 5000)
#println("$(trace[:intended_state]) \n$(trace[:intended_city]) \n$(trace[:intended_zip]) \n$(trace[:rent])")
#println("Log-likelihood: $lml_est")
