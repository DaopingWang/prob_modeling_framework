cd(dirname(@__FILE__))
include("modeling/embedded_correlation_lib.jl")
include("modeling/correlation_lib.jl")
include("modeling/util.jl")
using SQLite
using CSV
using DataFrames: DataFrame, Missing
using Statistics, LinearAlgebra, Logging, Random
using Gen

@gen function enriched_rent_emb_model(df, emb_df, emb_dict)
    @info "-----------------EMB"
    #state = @trace(occurrence(df, :state), :state)
    states = unique(df.state)
    occurrence = [sum(df.state .== s) for s in states]
    probs = LinearAlgebra.normalize(occurrence, 1)
    state = @trace(categorical_named(states, probs), :state => :realization)
    @info "$state"

    city = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:state,],
                                            ["embedding"],
                                            :city,
                                            "categorical",
                                            [state],
                                            10), :city)
    @info "$city"


    zip = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:state, :city],
                                            ["embedding", "embedding"],
                                            :zip,
                                            "categorical",
                                            [state, city],
                                            10), :zip)
    @info "$zip"

    ls_low = minimum(df.living_space)
    ls_high = maximum(df.living_space)
    living_space = @trace(uniform_continuous(ls_low, ls_high), :living_space => :realization)
    @info "$living_space"

    construct_year = @trace(uniform_categorical(df.construct_year), :construct_year => :realization)
    @info "$construct_year"

    heating_type = @trace(embedding_co_occurrence(df,
                                                    emb_df,
                                                    emb_dict,
                                                    [:construct_year, :city],
                                                    ["embedding", "embedding"],
                                                    :heating_type,
                                                    "categorical",
                                                    [construct_year, city],
                                                    10), :heating_type)
    @info "$heating_type"

    prob_parking = sum(df.has_parking) / size(df, 1)
    has_parking = @trace(bernoulli(prob_parking), :has_parking  => :realization)
    @info "Has parking? $has_parking"

    prob_balcony = sum(df.has_balcony) / size(df, 1)
    has_balcony = @trace(bernoulli(prob_balcony), :has_balcony  => :realization)
    @info "Has balcony? $has_balcony"
#=
    total_rent = @trace(numerical_co_occurrence(df,
                                                [:living_space, :state, :city, :has_parking, :has_balcony, :heating_type],
                                                ["numerical", "categorical", "categorical", "numerical", "numerical", "categorical"],
                                                (:rent, 15.),
                                                [(living_space, 10.), state, city, (has_parking, 0), (has_balcony, 0),
                                                heating_type],
                                                false,
                                                false), :rent)=#

    # TODO this is not working as finding neighbors does not incorperate numerical lhs attr correctly

    total_rent = @trace(embedding_co_occurrence(df,
                                                emb_df,
                                                emb_dict,
                                                [:living_space, :state, :city, :has_parking, :has_balcony, :heating_type],
                                                ["numerical", "embedding", "embedding", "numerical", "numerical", "embedding"],
                                                :rent,
                                                "numerical",
                                                [(living_space, 10.), state, city, (has_parking, 0), (has_balcony, 0), heating_type],
                                                10), :rent)

    @info "total $total_rent," #lsr $living_space_rent, chlsr $city_heating_ls_rent"
end


#################################################################################
CAT_ATTRS = [:state, :city, :zip, :construct_year, :heating_type]
NUM_ATTRS = [:living_space, :has_parking, :has_balcony, :rent]
TSV_PATH_PREFIX = "../data/tsv/20191207_enriched_rent_data/50000_"
CAT_EMBEDDING_DICT = merge([read_tsv("$(TSV_PATH_PREFIX)$(cat_attr)_meta.tsv",
                            "$(TSV_PATH_PREFIX)$(cat_attr)_vec.tsv")
                            for cat_attr in CAT_ATTRS]...)

cd("../data/rent_data/")

db = SQLite.DB()
rent_table = CSV.File("enriched_rent_data_50000_int.csv") |> SQLite.load!(db, "rent_table")
realization_df = SQLite.Query(db, "SELECT * FROM rent_table") |> DataFrame

embedding_df = replace_with_emb(realization_df, CAT_ATTRS, NUM_ATTRS, CAT_EMBEDDING_DICT)

#traces = [Gen.simulate(enriched_rent_emb_model, (realization_df,embedding_df,CAT_EMBEDDING_DICT)) for _=1:10]

#=
for i in 1:5
    constraints = Gen.choicemap()
    for cat_attr in CAT_ATTRS
        constraints[cat_attr => :realization] = realization_df[i, cat_attr]
    end
    for num_attr in NUM_ATTRS
        constraints[num_attr => :realization] = realization_df[i, num_attr]
    end
    (trace, weight) = Gen.generate(enriched_rent_emb_model,
                                    (realization_df,embedding_df, CAT_EMBEDDING_DICT),
                                    constraints)
    println(get_score(trace))
end
=#

function do_inference(model, args, constraints, num_iter)
    (trace, lml_est) = Gen.importance_resampling(model, args, constraints, num_iter)
    return (trace, lml_est)
end

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
