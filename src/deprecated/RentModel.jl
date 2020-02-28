include("deprecated/Util.jl")
include("modeling/util.jl")

using Gen
using DataFrames
using StringDistances

struct ExtData
    state_population_df::DataFrames.DataFrame
    state_city_population_mean_rent_df::DataFrames.DataFrame
end

function init_ext_data(state_population_df, state_city_population_mean_rent_df)
    global ext_data
    ext_data = ExtData(state_population_df, state_city_population_mean_rent_df)
end

@gen function add_typo(intention::String, prob)
    if @trace(Gen.bernoulli(prob), :add_more_typo)
        # Which action
        action = @trace(Gen.uniform_discrete(1, 3), :action)
        chr = @trace(Gen.uniform_discrete(1, 26), :chr)
        if action == 1
            # insertion
            pos = @trace(Gen.uniform_discrete(1, length(intention) + 1), :pos)
        else
            # deletion or replacement
            pos = @trace(Gen.uniform_discrete(1, length(intention)), :pos)
        end
        dirty_intention = get_word_with_typo(intention, action, pos, chr)
        return @trace(add_typo(dirty_intention, prob / 2.), :next_typo)
    else
        return intention
    end
end

function get_edit_score(string1::String, string2::String)
    return compare(string1, string2, TokenMax(Levenshtein()))
end

function get_edit_score(string_list::Array{String}, string::String)
    return [compare(i, string, TokenMax(Levenshtein()))  for i in string_list]
end

function get_weighted_prob_bef_norm(states_or_cities::Array{String}, populations, realization; mask=nothing)
    edit_dist_score = [1. for _=1:length(states_or_cities)]
    if realization != nothing
        edit_dist_score = get_edit_score(states_or_cities, realization)
    end
    edit_dist_score = [if i > 0.5 i * 5 else i end for i in edit_dist_score]
    #weighted_pop = populations .* edit_dist_score
    weighted_pop = edit_dist_score

    if mask != nothing
        weighted_pop[mask] .= 0.
    end
    return weighted_pop #./ sum(weighted_pop)
end

@gen function rent_model(realized_state, realized_city, realized_zip)
    # fetch external data
    global ext_data
    states_df = ext_data.state_population_df
    cities_df = ext_data.state_city_population_mean_rent_df

    # CHOOSE STATE PROPORTIONALLY TO POPULATION AND NUMBER OF TYPOS
    states_weights = get_weighted_prob_bef_norm(convert(Array{String}, states_df[:, :state]), states_df[:, :population], realized_state)
    states_prob = states_weights ./ sum(states_weights)
    intended_state_ind = @trace(Gen.categorical(states_prob), :intended_state_ind)
    intended_state = states_df[intended_state_ind, :state]

    # CHOOSE CITY AND ZIP PROPORTIONALLY TO POPULATION AND NUMBER OF TYPOS
    mask = cities_df[:state] .!= states_df[intended_state_ind, :state]
    cities_weights = get_weighted_prob_bef_norm(convert(Array{String}, cities_df[:, :city]), cities_df[:, :population], realized_city, mask=mask)
    if realized_zip != nothing
        cities_weights .*= get_edit_score([string(z) for z in cities_df[:, :zip]], string(realized_zip))
    end
    cities_prob = cities_weights ./ sum(cities_weights)
    intended_city_ind = @trace(Gen.categorical(cities_prob), :intended_city_ind)
    intended_city = cities_df[intended_city_ind, :city]
    intended_zip = cities_df[intended_city_ind, :zip]

    # SAMPLE RENT
    rent = @trace(Gen.normal(cities_df[intended_city_ind, :mean_rent], 2.), :rent)

    # ADD TYPOS
    #=
    if realized_city == nothing
        realized_city = @trace(add_typo(cities_df[intended_city_ind, :city], 0.2), :realized_city)
    end
    if realized_state == nothing
        realized_state = @trace(add_typo(cities_df[intended_city_ind, :state], 0.2), :realized_state)
    end
    if realized_zip == nothing
        realized_zip = @trace(add_typo(string(cities_df[intended_city_ind, :zip]), 0.2), :realized_zip)
    end
    =#
    println("\n$intended_state \n$intended_city \n$intended_zip \n$rent")
end

init_ext_data(read_csv(joinpath(@__DIR__, "../data/states.csv")), read_csv(joinpath(@__DIR__, "../data/cities.csv")))

# SIMULATION
# traces = [Gen.simulate(rent_model, ("hessen", "frankfurtammain", "60300")) for _=1:20]
# for trace in traces
    #println([cities_df[trace[:intended_city_ind], :state], cities_df[trace[:intended_city_ind], :city], trace[:rent]])
    # println(get_choices(trace))
# end

# println(states_df[!, :state])

# ERROR DETECTION
# setup constraints == "clean cells"
constraints = Gen.choicemap()

# conflicting constraints will result in -inf weight
# constraints[:city_ind] = 6
#constraints[:intended_state_ind] = 5 # Hessen
constraints[:rent] = 13.    # rent = 4EUR/m2
#constraints[:intended_city_ind] = 60
#constraints[:realized_zip] = "80332"

# generate one sample, evaluate its probability
for _=1:10
    (trace, weight) = Gen.generate(rent_model, ("Bayern", "mm", "8033"), constraints)
    println(weight)
end
#println([cities_df[trace[:city_ind], :state], cities_df[trace[:city_ind], :city], trace[:rent]])
#println(Gen.get_choices(trace))
#println(weight)

# REPAIR SUGGESTION
function do_inference(model, args, constraints, num_iter)
    (trace, lml_est) = Gen.importance_resampling(rent_model, args, constraints, num_iter)
    return (trace, lml_est)
end

#trace, lml_est = do_inference(rent_model, ("hessen", nothing, "60306"), constraints, 5000)
#println([ext_data.state_city_population_mean_rent_df[trace[:intended_city_ind], :state],
#ext_data.state_city_population_mean_rent_df[trace[:intended_city_ind], :city],
#ext_data.state_city_population_mean_rent_df[trace[:intended_city_ind], :zip],
#trace[:rent]])
#println(lml_est)
