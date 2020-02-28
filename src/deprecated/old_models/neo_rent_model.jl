cd(dirname(@__FILE__))
include("modeling/embedded_correlation_lib.jl")
include("modeling/correlation_lib.jl")
include("modeling/util.jl")
include("deprecated/Util.jl")
using SQLite
using CSV
using DataFrames: DataFrame, Missing
using Statistics, LinearAlgebra, Logging, Random
using Gen

@gen function neo_rent_plain_model(df)
    @info "-----------------NEOPLAIN"
    # w/o frequency info
    states = unique(df.state)
    occurrence = [sum(df.state .== s) for s in states]
    probs = LinearAlgebra.normalize(occurrence, 1)
    state = @trace(categorical_named(states, probs), :state => :realization)

    # with frequency info
    # state = uniform_categorical(states)
    @info "$state"

    city = @trace(categorical_co_occurrence(df,
                                            [:state,],
                                            ["categorical"],
                                            :city,
                                            [state],
                                            true), :city)

    @info "$city"

    zip = @trace(categorical_co_occurrence(df,
                                            [:state, :city],
                                            ["categorical", "categorical"],
                                            :zip,
                                            [state, city],
                                            true), :zip)
    @info "$zip"

    living_space = @trace(numerical_co_occurrence(df,
                                                    [:state, :city],
                                                    ["categorical", "categorical"],
                                                    :living_space,
                                                    [state, city],
                                                    false,
                                                    true), :living_space)
    @info "$living_space"

    construct_year = @trace(categorical_co_occurrence(df,
                                                    [:city, :living_space],
                                                    ["categorical", "numerical"],
                                                    :construct_year,
                                                    [city, (living_space, 10.)],
                                                    true), :construct_year)
    @info "$construct_year"

    total_rent = @trace(numerical_co_occurrence(df,
                                                [:living_space, :state, :city, :construct_year],
                                                ["numerical", "categorical", "categorical", "categorical"],
                                                :rent,
                                                [(living_space, 10.), state, city, construct_year],
                                                false,
                                                true), :rent)

    @info "Totally $total_rent"
end

@gen function neo_rent_emb_model(df, emb_df, emb_dict)
    @info "-----------------NEOEMB"
    # w/o frequency info
    states = unique(df.state)
    occurrence = [sum(df.state .== s) for s in states]
    probs = LinearAlgebra.normalize(occurrence, 1)
    state = @trace(categorical_named(states, probs), :state => :realization)

    # with frequency info
    # state = uniform_categorical(states)
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
                                            [:state, :city],
                                            ["embedding", "embedding"],
                                            :zip,
                                            "categorical",
                                            [state, city],
                                            zip_neighborhood_size), :zip)
    @info "$zip"

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
    @info "$living_space"

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
    @info "$construct_year"

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

    @info "Totally $total_rent" #lsr $living_space_rent, chlsr $city_heating_ls_rent"
end


#################################################################################
disable_logging(LogLevel(10))
CAT_ATTRS = [:state, :city, :construct_year, :zip, ]
NUM_ATTRS = [:living_space, :rent]
TSV_PATH_PREFIX = "../data/tsv/20200227_RA/rent_100_1024_1000_"
CAT_EMBEDDING_DICT = merge([read_tsv("$(TSV_PATH_PREFIX)$(cat_attr)_meta.tsv",
                            "$(TSV_PATH_PREFIX)$(cat_attr)_vec.tsv")
                            for cat_attr in CAT_ATTRS]...)

db = SQLite.DB()
#rent_table = CSV.File("../data/rent_data/neo_enriched_rent_30_per_city.csv") |> SQLite.load!(db, "rent_table")
rent_table = CSV.File("../data/tsv/20200227_RA/3000_k_1_p_5.csv") |> SQLite.load!(db, "rent_table")
df = SQLite.Query(db, "SELECT * FROM rent_table") |> DataFrame

emb_df = replace_with_emb(df, CAT_ATTRS, NUM_ATTRS, CAT_EMBEDDING_DICT)

#traces = [Gen.simulate(neo_rent_plain_model, (realization_df,)) for _=1:10]
#traces = [Gen.simulate(neo_rent_emb_model, (realization_df,embedding_df,CAT_EMBEDDING_DICT)) for _=1:10]
#=
for i in 1:10
    constraints = Gen.choicemap()
    for cat_attr in CAT_ATTRS
        constraints[cat_attr => :realization] = realization_df[i, cat_attr]
    end
    for num_attr in NUM_ATTRS
        constraints[num_attr => :realization] = realization_df[i, num_attr]
    end
    (trace, weight) = Gen.generate(neo_rent_plain_model,
                                    (realization_df, ),# embedding_df, CAT_EMBEDDING_DICT),
                                    constraints)
    println(get_score(trace))
end

(emb_traces, emb_scores) = k_most_improbable_neo(5,
                                    realization_df,
                                    realization_df,
                                    vcat(CAT_ATTRS, NUM_ATTRS),
                                    neo_rent_emb_model,
                                    (realization_df, embedding_df, CAT_EMBEDDING_DICT))

(plain_traces, plain_scores) = k_most_improbable_neo(5,
                                    realization_df,
                                    realization_df,
                                    vcat(CAT_ATTRS, NUM_ATTRS),
                                    neo_rent_plain_model,
                                    (realization_df, ))

for (i, trace) in enumerate(plain_traces)
    println("---------------")
    for attr in vcat(CAT_ATTRS, NUM_ATTRS)
        println("$(attr): $(trace[attr => :realization])")
    end
    println(scores[i])
end

function do_inference(model, args, constraints, num_iter)
    (trace, lml_est) = Gen.importance_resampling(model, args, constraints, num_iter)
    return (trace, lml_est)
end
=#
# Bayern,Wuerzburg,97070,59.61,1952,Oil,1,0,1366.83
#=
constraint = Gen.choicemap()
constraint[:state => :realization] = "Bayern"
#constraint[:city => :realization] = "Muenchen"
constraint[:living_space => :realization] = 59.
constraint[:construct_year => :realization] = 2000
constraint[:has_parking => :realization] = true
constraint[:rent => :realization] = 1366

trace, lml_est = do_inference(enriched_rent_model_plain,
                                (realization_df,),
                                constraint, 50)
println("$(trace[:zip => :realization]) \n$(trace[:has_balcony => :realization]) \n$(trace[:heating_type => :realization])")
println("city: $(trace[:city => :realization])")
println("ls: $(trace[:living_space => :realization])")
println("Log-likelihood: $lml_est")
=#

#disable_logging(LogLevel(10))

#precision recall using negsamp
k = sum(df.negative_sample)
(emb_traces, emb_scores, emb_ns, e_r, scores_emb) = precision_recall(k,
                                    df,
                                    df,
                                    vcat(CAT_ATTRS, NUM_ATTRS),
                                    neo_rent_emb_model,
                                    (df, emb_df, CAT_EMBEDDING_DICT))
emb_sorted_score_df, emb_ns = k_most_improbable(k,
                                            df,
                                            vcat(CAT_ATTRS, NUM_ATTRS),
                                            neo_rent_emb_model,
                                            (df, emb_df, CAT_EMBEDDING_DICT))
emb_precision = convert(Float32, sum(emb_ns)) / convert(Float32, k)
println("---------------")
println("Recall with embedding:")
println(round(e_r; digits=2))

println("k most improb:")
println(round(emb_precision; digits=2))

println("---------------")


(plain_traces, plain_scores, plain_ns, p_r, scores_plain) = precision_recall(k,
                                    df,
                                    df,
                                    vcat(CAT_ATTRS, NUM_ATTRS),
                                    neo_rent_plain_model,
                                    (df, ))

println("---------------")
println("Recall without embedding:")
println(round(p_r, digits=2))
println("---------------")
