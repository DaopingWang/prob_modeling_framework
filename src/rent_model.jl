cd(dirname(@__FILE__))
include("modeling/embedded_correlation_lib.jl") # backend generation functions
include("modeling/correlation_lib.jl") # backend generation functions
include("modeling/util.jl")
include("custom_proposal/neighborhood_proposal.jl")
using SQLite
using CSV
using DataFrames: DataFrame, Missing
using Statistics, LinearAlgebra, Logging, Random
using Gen

# All columns except target column are sampled from a uniform distribution.
# The target column values is then sampled based on co_occurrence of the sampled values.
# No prior knowledge on correlation between column attributes is given to the model.
@gen function rent_plain_no_rules_model(df::DataFrame)
    states = unique(df.state)
    state = @trace(uniform_categorical(states), :state => :realization)

    cities = unique(df.city)
    city = @trace(uniform_categorical(cities), :city => :realization)

    zips = unique(df.zip)
    zip = @trace(uniform_categorical(zips), :zip => :realization)

    living_space = @trace(uniform_continuous(minimum(df.living_space), maximum(df.living_space)), :living_space => :realization)

    construct_years = unique(df.construct_year)
    construct_year = @trace(uniform_categorical(construct_years), :construct_year => :realization)

    rent = @trace(numerical_co_occurrence(df,
                                            [:living_space, :state, :city, :zip, :construct_year],
                                            ["numerical", "categorical", "categorical", "categorical", "categorical"],
                                            :rent,
                                            [(living_space, 10.), state, city, zip, construct_year],
                                            false,
                                            true), :rent)
end

# generates PDB model for rental apartments dataset, without ICs
@gen function rent_emb_no_rules_model(df::DataFrame, emb_df::DataFrame, emb_dict::Dict)
    states = unique(df.state)
    state = @trace(uniform_categorical(states), :state => :realization)

    cities = unique(df.city)
    city = @trace(uniform_categorical(cities), :city => :realization)

    zips = unique(df.zip)
    zip = @trace(uniform_categorical(zips), :zip => :realization)

    living_space = @trace(uniform_continuous(minimum(df.living_space), maximum(df.living_space)), :living_space => :realization)

    construct_years = unique(df.construct_year)
    construct_year = @trace(uniform_categorical(construct_years), :construct_year => :realization)

    rent_neighborhood_size = 15
    rent = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:living_space, :state, :city, :zip, :construct_year],
                                            ["numerical", "embedding", "embedding", "embedding", "embedding"],
                                            :rent,
                                            "numerical",
                                            [(living_space, 10.), state, city, zip, construct_year],
                                            rent_neighborhood_size), :rent)
end

# generates PDB model for rental apartments dataset, without embeddings
@gen function rent_plain_model(df::DataFrame)
    states = unique(df.state)
    state = @trace(uniform_categorical(states), :state => :realization)

    #@info "$state"

    city = @trace(categorical_co_occurrence(df,
                                            [:state,],
                                            ["categorical"],
                                            :city,
                                            [state],
                                            true), :city)

    #@info "$city"

    zip = @trace(categorical_co_occurrence(df,
                                            [:state, :city],
                                            ["categorical", "categorical"],
                                            :zip,
                                            [state, city],
                                            true), :zip)
    #@info "$zip"

    living_space = @trace(numerical_co_occurrence(df,
                                                    [:state, :city],
                                                    ["categorical", "categorical"],
                                                    :living_space,
                                                    [state, city],
                                                    false,
                                                    true), :living_space)
    #@info "$living_space"

    construct_year = @trace(categorical_co_occurrence(df,
                                                    [:city, :living_space],
                                                    ["categorical", "numerical"],
                                                    :construct_year,
                                                    [city, (living_space, 10.)],
                                                    true), :construct_year)
    #@info "$construct_year"

    total_rent = @trace(numerical_co_occurrence(df,
                                                [:living_space, :state, :city, :construct_year],
                                                ["numerical", "categorical", "categorical", "categorical"],
                                                :rent,
                                                [(living_space, 10.), state, city, construct_year],
                                                false,
                                                true), :rent)

    #@info "Totally $total_rent"
end

# generates PDB model for rental apartments dataset, with embeddings
# O(n_dependencies n_obs log n_obs)
@gen function rent_emb_model(df::DataFrame, emb_df::DataFrame, emb_dict::Dict)
    states = unique(df.state)
    state = @trace(uniform_categorical(states), :state => :realization)

    #@info "$state"

    # sample city from neighborhood, this is a hyperparameter...
    city_neighborhood_size = 1
    city = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:state,],
                                            ["embedding"],
                                            :city,
                                            "categorical",
                                            [state],
                                            city_neighborhood_size), :city)
    #@info "$city"

    # neighborhood size == 1 means we want no emb based neighbors
    zip_neighborhood_size = 1
    zip = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:state, :city],
                                            ["embedding", "embedding"],
                                            :zip,
                                            "categorical",
                                            [state, city],
                                            zip_neighborhood_size), :zip)
    #@info "$zip"

    ls_neighborhood_size = 15
    living_space = @trace(embedding_co_occurrence(df,
                                                    emb_df,
                                                    emb_dict,
                                                    [:state, :city],
                                                    ["embedding", "embedding"],
                                                    :living_space,
                                                    "numerical",
                                                    [state, city],
                                                    ls_neighborhood_size), :living_space)
    #@info "$living_space"

    cy_neighborhood_size = 15
    construct_year = @trace(embedding_co_occurrence(df,
                                                    emb_df,
                                                    emb_dict,
                                                    [:city, :living_space],
                                                    ["embedding", "numerical"],
                                                    :construct_year,
                                                    "categorical",
                                                    [city, (living_space, 30)],
                                                    cy_neighborhood_size), :construct_year)
    #@info "$construct_year"

    rent_neighborhood_size = 15
    total_rent = @trace(embedding_co_occurrence(df,
                                                emb_df,
                                                emb_dict,
                                                [:living_space, :state, :city, :construct_year],
                                                ["numerical", "embedding", "embedding", "embedding"],
                                                :rent,
                                                "numerical",
                                                [(living_space, 10.), state, city, construct_year],
                                                rent_neighborhood_size), :rent)
end

#################################################################################
disable_logging(LogLevel(10))
# Categorical attributes replaced with embeddings
CAT_ATTRS = [:state, :city, :zip, :construct_year]
# numerical attributes
NUM_ATTRS = [:living_space, :rent]

# Path of embedding files and csv
TSV_PATH_PREFIX = "../data/rent_data/20200227_showcase/rent_100_1024_1000_"
CSV_FILE = "../data/rent_data/20200227_showcase/3000_k_2_p_0.2.csv"

CAT_EMB_DICT = merge([read_tsv("$(TSV_PATH_PREFIX)$(cat_attr)_meta.tsv",
                            "$(TSV_PATH_PREFIX)$(cat_attr)_vec.tsv")
                            for cat_attr in CAT_ATTRS]...)
EMB_CAT_DICT = Dict(value => key for (key, value) in CAT_EMB_DICT)

db = SQLite.DB()
rent_table = CSV.File(CSV_FILE) |> SQLite.load!(db, "rent_table")
df = SQLite.Query(db, "SELECT * FROM rent_table") |> DataFrame

emb_df = replace_with_emb(df, CAT_ATTRS, NUM_ATTRS, CAT_EMB_DICT)

####################
# precision recall with negative samples
k = sum(df.negative_sample)
@time emb_sorted_score_df, emb_ns = k_most_improbable(k,
                                            df,
                                            vcat(CAT_ATTRS, NUM_ATTRS),
                                            rent_emb_model,
                                            (df, emb_df, CAT_EMB_DICT))
emb_precision = convert(Float32, sum(emb_ns)) / convert(Float32, k)
@time emb_NR_sorted_score_df, emb_NR_ns = k_most_improbable(k,
                                            df,
                                            vcat(CAT_ATTRS, NUM_ATTRS),
                                            rent_emb_no_rules_model,
                                            (df, emb_df, CAT_EMB_DICT))
emb_NR_precision = convert(Float32, sum(emb_NR_ns)) / convert(Float32, k)


@time plain_sorted_score_df, plain_ns = k_most_improbable(k,
                                            df,
                                            vcat(CAT_ATTRS, NUM_ATTRS),
                                            rent_plain_model,
                                            (df, ))
plain_precision = convert(Float32, sum(plain_ns)) / convert(Float32, k)
@time plain_NR_sorted_score_df, plain_NR_ns = k_most_improbable(k,
                                            df,
                                            vcat(CAT_ATTRS, NUM_ATTRS),
                                            rent_plain_no_rules_model,
                                            (df, ))
plain_NR_precision = convert(Float32, sum(plain_NR_ns)) / convert(Float32, k)
println("---------------")
println("Rent model. Precision with embedding:")
println(round(emb_precision; digits=3))
println("Rent model. Precision with embedding, no rules:")
println(round(emb_NR_precision; digits=3))
println("---------------")
println("---------------")
println("Rent model. Precision without embedding:")
println(round(plain_precision, digits=3))
println("Rent model. Precision without embedding, no rules:")
println(round(plain_NR_precision, digits=3))
println("---------------")

#################################################
# Write PR and ROC curves
#=
emb_pr_df, emb_roc_df = get_pr_roc(emb_sorted_score_df, k)
CSV.write("ra_emb_roc.csv", emb_roc_df)
CSV.write("ra_emb_pr.csv", emb_pr_df)
emb_NR_pr_df, emb_NR_roc_df = get_pr_roc(emb_NR_sorted_score_df, k)
CSV.write("ra_emb_NR_roc.csv", emb_NR_roc_df)
CSV.write("ra_emb_NR_pr.csv", emb_NR_pr_df)
plain_pr_df, plain_roc_df = get_pr_roc(plain_sorted_score_df, k)
CSV.write("ra_plain_roc.csv", plain_roc_df)
CSV.write("ra_plain_pr.csv", plain_pr_df)
plain_NR_pr_df, plain_NR_roc_df = get_pr_roc(plain_NR_sorted_score_df, k)
CSV.write("ra_plain_NR_roc.csv", plain_NR_roc_df)
CSV.write("ra_plain_NR_pr.csv", plain_NR_pr_df)=#

#################################################
# Inference
#=
hessen_bayreuth = df[3171, :]
trace = do_neighborhood_inference(rent_emb_model,
                                (df, emb_df, CAT_EMB_DICT),
                                :state,
                                emb_df,
                                CAT_EMB_DICT,
                                EMB_CAT_DICT,
                                hessen_bayreuth,
                                5,
                                50)
println(get_trace_vals(trace, vcat(CAT_ATTRS, NUM_ATTRS)))
=#
