cd(dirname(@__FILE__))
include("modeling/embedded_correlation_lib.jl")
include("modeling/correlation_lib.jl")
include("modeling/util.jl")

using SQLite
using Gen
using CSV
using DataFrames: DataFrame, Missing
using LinearAlgebra
using Statistics
using Logging
using Random

@gen function simple_rent_plain_NN_model(df)
    states = unique(df.state)
    state = @trace(uniform_categorical(states), :state => :realization)

    cities = unique(df.city)
    city = @trace(uniform_categorical(cities), :city => :realization)

    zips = unique(df.zip)
    zip = @trace(uniform_categorical(zips), :zip => :realization)

    rent = @trace(numerical_co_occurrence(df,
                                            [:state, :city, :zip],
                                            ["categorical", "categorical", "categorical"],
                                            :rent,
                                            [state, city, zip],
                                            false,
                                            true), :rent)
end

@gen function simple_rent_emb_NN_model(df, emb_df, emb_dict)
    states = unique(df.state)
    state = @trace(uniform_categorical(states), :state => :realization)

    cities = unique(df.city)
    city = @trace(uniform_categorical(cities), :city => :realization)

    zips = unique(df.zip)
    zip = @trace(uniform_categorical(zips), :zip => :realization)

    rent_neighborhood_size = 5
    rent = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:state, :city, :zip],
                                            ["embedding", "embedding", "embedding"],
                                            :rent,
                                            "numerical",
                                            [state, city, zip],
                                            rent_neighborhood_size), :rent)
end

@gen function simple_rent_plain_model(df)
    @info "-----------------SIMPLEPLAIN"
    # w frequency info
    states = unique(df.state)
    #occurrence = [sum(df.state .== s) for s in states]
    #probs = LinearAlgebra.normalize(occurrence, 1)
    #state = @trace(categorical_named(states, probs), :state => :realization)

    # wo frequency info
    state = @trace(uniform_categorical(states), :state => :realization)
    @info "$state"

    city = @trace(categorical_co_occurrence(df,
                                            [:state,],
                                            ["categorical"],
                                            :city,
                                            [state],
                                            true), :city)

    @info "$city"

    zip = @trace(categorical_co_occurrence(df,
                                            [:city],
                                            ["categorical"],
                                            :zip,
                                            [city],
                                            true), :zip)
    @info "$zip"

    # Here if I set rent to be conditioned on zip, recall would be 0 as zip way too sparse
    total_rent = @trace(numerical_co_occurrence(df,
                                                [:city],
                                                ["categorical"],
                                                :rent,
                                                [city],
                                                false,
                                                true), :rent)

    @info "Totally $total_rent"
end

@gen function simple_rent_emb_model(df, emb_df, emb_dict)
    @info "-----------------SIMPLEEMB"
    # w frequency info
    states = unique(df.state)
    #occurrence = [sum(df.state .== s) for s in states]
    #probs = LinearAlgebra.normalize(occurrence, 1)
    #state = @trace(categorical_named(states, probs), :state => :realization)

    # wo frequency info
    state = @trace(uniform_categorical(states), :state => :realization)
    @info "$state"

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
    @info "$city"

    # neighborhood size == 1 means we want no emb based neighbors
    zip_neighborhood_size = 1
    zip = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:city],
                                            ["embedding"],
                                            :zip,
                                            "categorical",
                                            [city],
                                            zip_neighborhood_size), :zip)
    @info "$zip"

    rent_neighborhood_size = 5
    total_rent = @trace(embedding_co_occurrence(df,
                                                emb_df,
                                                emb_dict,
                                                [:zip],
                                                ["embedding"],
                                                :rent,
                                                "numerical",
                                                [zip],
                                                rent_neighborhood_size), :rent)

    @info "Totally $total_rent" #lsr $living_space_rent, chlsr $city_heating_ls_rent"
end

CAT_ATTRS = [:state, :city, :zip]
NUM_ATTRS = [:rent]
#TSV_PATH_PREFIX = "../data/tsv/20191218_simple_5_oversampling/num_zip/simple_5_os_"
#RENT_FILE_PATH = "../data/rent_data/simple_rent_5_per_city.csv"
TSV_PATH_PREFIX = "../data/tsv/20191230_NgNEGSAMP_simple_20/cat_zip/NgNEGSAMP_simple_20_"
RENT_FILE_PATH = "../data/rent_data/NEGSAMP_simple_rent_20_per_city.csv"

CAT_EMBEDDING_DICT = merge([read_tsv("$(TSV_PATH_PREFIX)$(cat_attr)_meta.tsv",
                            "$(TSV_PATH_PREFIX)$(cat_attr)_vec.tsv")
                            for cat_attr in CAT_ATTRS]...)

db = SQLite.DB()
rent_table = CSV.File(RENT_FILE_PATH) |> SQLite.load!(db, "rent_table")
df = SQLite.Query(db, "SELECT * FROM rent_table") |> DataFrame

emb_df = replace_with_emb(df, CAT_ATTRS, NUM_ATTRS, CAT_EMBEDDING_DICT)

#traces = [Gen.simulate(simple_rent_plain_model, (df,)) for _=1:10]
#traces = [Gen.simulate(simple_rent_emb_model, (df,emb_df,CAT_EMBEDDING_DICT)) for _=1:10]
#=
(emb_traces, emb_scores, emb_ns, _) = precision_recall(5,
                                    df,
                                    df,
                                    vcat(CAT_ATTRS, NUM_ATTRS),
                                    simple_rent_emb_model,
                                    (df, emb_df, CAT_EMBEDDING_DICT))

(plain_traces, plain_scores, plain_ns, _) = precision_recall(5,
                                    df,
                                    df,
                                    vcat(CAT_ATTRS, NUM_ATTRS),
                                    simple_rent_plain_model,
                                    (df, ))

for (i, trace) in enumerate(plain_traces)
    println("---------------")
    for attr in vcat(CAT_ATTRS, NUM_ATTRS)
        println("$(attr): $(trace[attr => :realization])")
    end
    println(plain_scores[i])
    println("NS: $(plain_ns[i])")
end

for (i, trace) in enumerate(emb_traces)
    println("---------------emb")
    for attr in vcat(CAT_ATTRS, NUM_ATTRS)
        println("$(attr): $(trace[attr => :realization])")
    end
    println(emb_scores[i])
    println("NS: $(emb_ns[i])")
end=#


#precision recall using negsamp
k = sum(df.negative_sample)
(emb_traces, emb_scores, emb_ns, e_r, scores_emb) = precision_recall(k,
                                    df,
                                    df,
                                    vcat(CAT_ATTRS, NUM_ATTRS),
                                    simple_rent_emb_model,
                                    (df, emb_df, CAT_EMBEDDING_DICT))
(emb_NN_traces, emb_NN_scores, emb_NN_ns, e_NN_r, scores_emb_NN) = precision_recall(k,
                                    df,
                                    df,
                                    vcat(CAT_ATTRS, NUM_ATTRS),
                                    simple_rent_emb_NN_model,
                                    (df, emb_df, CAT_EMBEDDING_DICT))

println("---------------")
println("Simple Recall with embedding:")
println(round(e_r; digits=2))
println("Simple Recall with embedding, NN:")
println(round(e_NN_r; digits=2))
println("---------------")
#=
for (i, trace) in enumerate(emb_traces)
    if i > 10
        continue
    end
    println("---------------emb")
    for attr in vcat(CAT_ATTRS, NUM_ATTRS)
        println("$(attr): $(trace[attr => :realization])")
    end
    println(emb_scores[i])
    println(emb_ns[i])
end=#

(plain_traces, plain_scores, plain_ns, p_r, scores_plain) = precision_recall(k,
                                    df,
                                    df,
                                    vcat(CAT_ATTRS, NUM_ATTRS),
                                    simple_rent_plain_model,
                                    (df, ))
(plain_NN_traces, plain_NN_scores, plain_NN_ns, p_NN_r, scores_NN_plain) = precision_recall(k,
                                    df,
                                    df,
                                    vcat(CAT_ATTRS, NUM_ATTRS),
                                    simple_rent_plain_NN_model,
                                    (df, ))

println("---------------")
println("Simple rent model. Recall without embedding:")
println(round(p_r, digits=2))
println("Simple rent model. Recall without embedding, NN:")
println(round(p_NN_r, digits=2))
println("---------------")
#=
for (i, trace) in enumerate(plain_traces)
    if i > 10
        continue
    end
    println("---------------plain")
    for attr in vcat(CAT_ATTRS, NUM_ATTRS)
        println("$(attr): $(trace[attr => :realization])")
    end
    println(plain_scores[i])
    println(plain_ns[i])
end=#
