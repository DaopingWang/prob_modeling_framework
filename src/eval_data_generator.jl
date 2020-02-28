cd(dirname(@__FILE__))
include("custom_distributions/custom_distribution_lib.jl")
include("modeling/util.jl")

using SQLite
using Gen
using StaticArrays
using CSV
using DataFrames: DataFrame
using LinearAlgebra

# negative sampler used for manual anomaly generation
function eval_negative_sampler(df::DataFrame, pivot::Symbol, attrs::Array, n::Int, p)
    attrs_values_unique = [unique(df[:, attr]) for attr in attrs]
    n_s_df_rows = []

    for i = 1:n
        sampled = false
        entry_i = uniform_discrete(1, size(df, 1))
        n_s_buffer = deepcopy(df[entry_i, :])
        for (j, attr) in enumerate(attrs)
            if bernoulli(p)
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
                n_s_buffer[attr] += 2000.
                #println(n_s_buffer[attr])
            end
        end
        if !sampled
            #println("not sampled")
            j = uniform_discrete(1, length(attrs))
            attr = attrs[j]
            pos_attr_vals = unique(df[df[:, pivot].==n_s_buffer[pivot], attr])
            neg_attr_vals = filter(x->x ∉ pos_attr_vals, attrs_values_unique[j])
            n_s_buffer[attr] = uniform_categorical(neg_attr_vals)
            if attr == :rent
                #println("get")
                n_s_buffer[attr] += 2000.
                #println(n_s_buffer[attr])
            end
        end
        push!(n_s_df_rows, deepcopy(n_s_buffer))
    end

    return n_s_df_rows
end

function generate_eval_rent_data(sampled_df, k, p)
    small_neo_df = deepcopy(sampled_df)
    orig_attrs = names(small_neo_df)
    small_neo_df.negative_sample = [0 for _=1:size(small_neo_df, 1)]
    kk = convert(Int, ceil(k*size(small_neo_df, 1)))
    #n_s_rows = negative_sampler(small_neo_df[:, orig_attrs], :city, [:living_space, :rent], 5)
    n_s_rows = eval_negative_sampler(small_neo_df[:, orig_attrs], :city, [:zip, :state, :rent], kk, p)
    n_s_df = DataFrame()
    for row in n_s_rows
        push!(n_s_df, row)
    end
    n_s_df.negative_sample = [1 for _=1:size(n_s_df, 1)]
    small_neo_df = vcat(small_neo_df, n_s_df)
    return small_neo_df
end

function generate_eval_ek_data(sampled_df, k, p)
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

@gen function rent_generation_model_given_city(intended_city::String)
    city_df = SQLite.Query(db, "SELECT
    state,
    zip,
    zip_start,
    zip_end,
    mean_rent,
    no_heating
    FROM truth_table WHERE city='$intended_city'") |> DataFrame

    intended_state = city_df.state[1]
    zip_start = city_df.zip_start[1]
    zip_end = city_df.zip_end[1]
    intended_zip = uniform_discrete(zip_start, zip_end)

    cy_df = SQLite.Query(db, "SELECT DISTINCT construct_year
    FROM rent_table WHERE city='$intended_city'") |> DataFrame
    num_cys = size(cy_df, 1)
    i = uniform_discrete(1, num_cys)
    intended_cy = cy_df.construct_year[i]

    mean_rent_df = SQLite.Query(db, "SELECT DISTINCT mean_rent
    FROM truth_table WHERE city='$intended_city'") |> DataFrame
    mean_rent_truth = mean_rent_df.mean_rent[1] + 2.5# - 2.

    mean_rent = mean_rent_truth

    living_space = uniform_continuous(20, 200)

    base_rent = normal(living_space * mean_rent, 10.)

    return (intended_state,
            intended_city,
            intended_zip,
            round(living_space; digits=2),
            intended_cy,
            round(base_rent; digits=2))
end

# sampler for generating the syntatic RA dataset

@gen function rent_generation_model()
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

    city_df = SQLite.Query(db, "SELECT DISTINCT city
    FROM rent_table WHERE state='$intended_state'") |> DataFrame
    num_cities = size(city_df, 1)
    i = uniform_discrete(1, num_cities)
    intended_city = city_df.city[i]

    zip_df = SQLite.Query(db, "SELECT DISTINCT zip
    FROM rent_table WHERE city='$intended_city'") |> DataFrame
    num_zips = size(zip_df, 1)
    i = uniform_discrete(1, num_zips)
    intended_zip = zip_df.zip[i]

    cy_df = SQLite.Query(db, "SELECT DISTINCT construct_year
    FROM rent_table WHERE city='$intended_city'") |> DataFrame
    num_cys = size(cy_df, 1)
    i = uniform_discrete(1, num_cys)
    intended_cy = cy_df.construct_year[i]

    mean_rent_df = SQLite.Query(db, "SELECT DISTINCT mean_rent
    FROM truth_table WHERE city='$intended_city'") |> DataFrame
    mean_rent_truth = mean_rent_df.mean_rent[1] + 2.5# - 2.

    observed_ls_rent_df = SQLite.Query(db, "SELECT living_space, rent
    FROM rent_table WHERE city='$intended_city'") |> DataFrame
    i = uniform_discrete(1, size(observed_ls_rent_df, 1))
    observed_ls = observed_ls_rent_df.living_space[i]
    observed_rent = observed_ls_rent_df.rent[i]

    mean_rent = mean_rent_truth
    if intended_cy < 1949
        mean_rent += 1.5
    elseif intended_cy < 1969
        mean_rent -= 1.
    elseif intended_cy > 1990
        mean_rent += (intended_cy-1990) * 0.25
    end

    # The smaller the appartement, the more expensive the unit rent gets
    living_space = uniform_continuous(20, 200)

    if living_space < 100
        mean_rent += (100 - living_space) * 0.0425
    else
        mean_rent -= (living_space - 100) * 0.0425
    end

    base_rent = living_space * mean_rent

    total_rent = base_rent# + additional_cost + heating_cost
    #total_rent = normal(total_rent, 20.)
    if total_rent < 100
        println("bad rent generated: "*string(total_rent))
        living_space = observed_ls
        total_rent = observed_rent
    end
    return (intended_state,
            intended_city,
            intended_zip,
            round(living_space; digits=2),
            intended_cy,
            round(total_rent; digits=2))
end

# load necessary external info for the RA dataset
db = SQLite.DB()
truth_table = CSV.File(joinpath(@__DIR__, "../data/cities.csv")) |> SQLite.load!(db, "truth_table")
rent_table = CSV.File(joinpath(@__DIR__, "../data/tsv/eval/clean_ra_dataset.csv")) |> SQLite.load!(db, "rent_table")
truth_df = SQLite.Query(db, "SELECT * FROM truth_table") |> DataFrame

####################### pdb parameters ############################

ss = [3000, 500, 6000, 10000] # |tuples(D)|
kk = [0.05, 0.01, 0.2, 0.4] # number of manual corrupted tuples = |tuples(D)| * k
pp = [0.2, 0.5, 0.8, 1.0] # probability of inducing error at each attribute

####################### rental apartments dataset ############################
for s in ss
    neo_df = DataFrame(state = String[],
                        city = String[],
                        zip = Int[],
                        living_space = Float32[],
                        construct_year = Int[],
                        rent = Float32[])
    cities = truth_df.city
    n_each_city = ceil(s / length(cities))
    for city in cities
        data = [rent_generation_model_given_city(city) for _=1:n_each_city]
        for e in data
            push!(neo_df, e)
        end
    end
    eval_df = generate_eval_rent_data(neo_df, 0.05, 0.5)
    file_name = string(s)*"_k_"*string(005)*"_p_"*string(05)*".csv"
    CSV.write(file_name, eval_df)
end

for k in kk
    neo_df = DataFrame(state = String[],
                        city = String[],
                        zip = Int[],
                        living_space = Float32[],
                        construct_year = Int[],
                        rent = Float32[])
    data = [rent_generation_model() for _=1:3000]
    for e in data
        push!(neo_df, e)
    end
    eval_df = generate_eval_rent_data(neo_df, k, 0.2)
    file_name = string(3000)*"_k_"*string(k)*"_p_"*string(02)*".csv"
    CSV.write(file_name, eval_df)
end

for p in pp
    neo_df = DataFrame(state = String[],
                        city = String[],
                        zip = Int[],
                        living_space = Float32[],
                        construct_year = Int[],
                        rent = Float32[])
    data = [rent_generation_model() for _=1:3000]
    for e in data
        push!(neo_df, e)
    end
    eval_df = generate_eval_rent_data(neo_df, 0.05, p)
    file_name = string(3000)*"_k_"*string(02)*"_p_"*string(p)*".csv"
    CSV.write(file_name, eval_df)
end

######################## mercateo dataset ###############################
CAT_ATTRS = [#:unit,
            :article_id,

            ]
EMB_ATTRS = [:catalog_id,
                #:lower_bound,
                :unit,
                :keywords,
                :manufacturer_name,
                #:ean,
                :set_id,
                ]
NUM_ATTRS = [:ek_amount,
                #:vk_amount,
                ]

all_attrs = vcat(CAT_ATTRS, vcat(EMB_ATTRS, NUM_ATTRS))
column_names = reduce(*, [String(attr) * ", " for attr in all_attrs])[1:end-2]
db = SQLite.DB()
ek_table = CSV.File("../data/mercateo/NEGSAMP_top_200_patched_syn_avg.csv") |> SQLite.load!(db, "ek_table")
good_df = SQLite.Query(db, "SELECT $column_names FROM ek_table WHERE negative_sample='0' ") |> DataFrame

for k in kk
    eval_df = generate_eval_ek_data(good_df, k, 0.2)
    file_name = "merc_k_"*string(k)*"_p_"*string(0.2)*".csv"
    CSV.write(file_name, eval_df)
end

for p in pp
    eval_df = generate_eval_ek_data(good_df, 0.2, p)
    file_name = "merc_k_"*string(0.05)*"_p_"*string(p)*".csv"
    CSV.write(file_name, eval_df)
end
