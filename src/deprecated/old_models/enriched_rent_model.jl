cd("C:/git/lets_learn/infclean/src/")
include("modeling/correlation_lib.jl")
include("modeling/util.jl")
using SQLite
using CSV
using DataFrames: DataFrame, Missing
using Statistics, LinearAlgebra, Logging, Random
using Gen

@gen function enriched_rent_model_plain(df)
    @info "-----------------PLAIN"
    #state = @trace(occurrence(df, :state), :state)
    states = unique(df.state)
    occurrence = [sum(df.state .== s) for s in states]
    probs = LinearAlgebra.normalize(occurrence, 1)
    state = @trace(categorical_named(states, probs), :state => :realization)
    @info "$state"

    city = @trace(categorical_co_occurrence(df,
                                            [:state,],
                                            ["categorical"],
                                            :city,
                                            [state],
                                            false), :city)
    @info "$city"


    zip = @trace(categorical_co_occurrence(df,
                                            [:state, :city],
                                            ["categorical", "categorical"],
                                            :zip,
                                            [state, city],
                                            false), :zip)
    @info "$zip"

    ls_low = minimum(df.living_space)
    ls_high = maximum(df.living_space)
    living_space = @trace(uniform_continuous(ls_low, ls_high), :living_space => :realization)
    @info "$living_space"

    construct_year = @trace(uniform_categorical(df.construct_year), :construct_year => :realization)
    @info "$construct_year"

    heating_type = @trace(categorical_co_occurrence(df,
                                                    [:construct_year, :city],
                                                    ["categorical", "categorical"],
                                                    :heating_type,
                                                    [construct_year, city],
                                                    false), :heating_type)
    @info "$heating_type"

    prob_parking = sum(df.has_parking) / size(df, 1)
    has_parking = @trace(bernoulli(prob_parking), :has_parking  => :realization)
    @info "Has parking? $has_parking"

    prob_balcony = sum(df.has_balcony) / size(df, 1)
    has_balcony = @trace(bernoulli(prob_balcony), :has_balcony  => :realization)
    @info "Has balcony? $has_balcony"

    total_rent = @trace(numerical_co_occurrence(df,
                                                [:living_space, :state, :city, :has_parking, :has_balcony, :heating_type],
                                                ["numerical", "categorical", "categorical", "numerical", "numerical", "categorical"],
                                                (:rent, 15.),
                                                [(living_space, 10.), state, city, (has_parking, 0), (has_balcony, 0),
                                                heating_type],
                                                false,
                                                false), :rent)

    @info "total $total_rent," #lsr $living_space_rent, chlsr $city_heating_ls_rent"
end


#################################################################################
cd("../data/rent_data/")

db = SQLite.DB()
rent_table = CSV.File("enriched_rent_data_50000_int.csv") |> SQLite.load!(db, "rent_table")
validation_table = CSV.File("enriched_rent_data_50000_int_validation.csv") |> SQLite.load!(db, "validation_table")
realization_df = SQLite.Query(db, "SELECT * FROM rent_table") |> DataFrame

cd("..")

traces = [Gen.simulate(enriched_rent_model_plain, (realization_df,)) for _=1:10]

#=
temp_attrs = Dict(:state => :state,
            :city => :city,
            :zip => :zip,
            :rent => :rent,
            :living_space => :living_space,
            :construct_year => :construct_year,
            :heating_type => :heating_type,
            :has_parking => :has_parking,
            :has_balcony => :has_balcony)
validation_df = SQLite.Query(db, "SELECT * FROM validation_table") |> DataFrame
# TODO bugfix
k_most_improbable_neo(10, realization_df, validation_df, temp_attrs, enriched_rent_model_plain)
=#
#=
for i in 1:5
    constraints = Gen.choicemap()
    for cat_attr in CAT_ATTRS
        constraints[cat_attr => :realization] = realization_df[i, cat_attr]
    end
    for num_attr in NUM_ATTRS
        constraints[num_attr => :realization] = realization_df[i, num_attr]
    end
    (trace, weight) = Gen.generate(enriched_rent_model_plain,
                                    (realization_df,),
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
