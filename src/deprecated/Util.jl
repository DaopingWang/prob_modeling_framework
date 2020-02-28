using CSV
using StringDistances

function read_csv(file_path::String)
    return CSV.File(file_path) |> DataFrame!
end

function word_to_distance_dict(word, word_list)
    return [w => StringDistances.evaluate(Levenshtein(), word, w) for w in word_list]
end

function get_word_with_typo(word::String, action, pos, chr)
    # Action: 1 for insertion, 2 for removal, 3 for replacement
    if action==1
        converted_chr = convert(Char, chr+96)
        return word[1:pos-1]*converted_chr*word[pos:end]
    elseif action==2
        return word[1:pos-1]*word[pos+1:end]
    else
        converted_chr = convert(Char, chr+96)
        return replace(word, word[pos]=>converted_chr)
    end
end


function negative_sampler(df, attr, ignore_attrs, n_s_per_unique_attr)
    attr_values_unique = unique(df[:, attr])
    other_attrs = filter(x->x!=attr, names(df))
    other_attrs = filter(x->x ∉ ignore_attrs, other_attrs)
    n_s_df_rows = []
    # for each city
    for (i, attr_val) in enumerate(attr_values_unique)
        # get all entries with current city val
        attr_val_df = df[df[:, attr].==attr_val, :]

        # samples 5 negative samples for this city, from the positive entry
        for j = 1:n_s_per_unique_attr
            attr_val_entry_i = uniform_discrete(1, size(attr_val_df, 1))
            #n_s_buffer = DataFrameRow(attr_val_df, 1)
            n_s_buffer = deepcopy(attr_val_df[attr_val_entry_i, :])
            # choose one attr to make dirty, e.g. state
            neg_attr = uniform_categorical(other_attrs)
            # which states occurred together with city?
            pos_attr_vals = unique(attr_val_df[:, neg_attr])
            # which have not?
            neg_attr_vals = unique(filter(x->x ∉ pos_attr_vals, df[:, neg_attr]))
            # pick one state that has not occurred together with city
            try
                n_s_buffer[neg_attr] = uniform_categorical(neg_attr_vals)
            catch e
                @warn "Chosen neg_attr ($neg_attr) has no neg_attr_vals. Retry."
                j -= 1
                continue
            end
            # for other attrs beside city and state, choose randomly whether they
            # are also to be changed

            for other_other_attr in filter(x->x!=neg_attr, other_attrs)
                try
                    pos_other_other_attr_vals = unique(attr_val_df[:, other_other_attr])
                    neg_other_other_attr_vals = unique(filter(x->x ∉ pos_other_other_attr_vals, df[:, other_other_attr]))
                    if bernoulli(0.0)
                        # add wrong val
                        n_s_buffer[other_other_attr] = uniform_categorical(neg_other_other_attr_vals)
                    #else
                        # add occurred val
                        #n_s_buffer[other_other_attr] = uniform_categorical(pos_other_other_attr_vals)
                    end
                catch e
                    @warn "neg_other_other_attr_vals for ($other_other_attr) is empty list. Skip."
                    continue
                end
            end
            #for ignore_attr in ignore_attrs
            #    pos_ignore_attr_vals = unique(attr_val_df[:, ignore_attr])
            #    n_s_buffer[ignore_attr] = uniform_categorical(pos_ignore_attr_vals)
            #end
            push!(n_s_df_rows, deepcopy(n_s_buffer))
        end
    end
    return n_s_df_rows
end



function precision_recall(k, observation_df, validation_df, attrs, model, args)
    @assert typeof(observation_df) == DataFrame
    println("")
    score_list = []
    k_trace_list = []
    k_score_list = []
    k_n_s_list = []
    for i in 1:size(validation_df, 1)
        #print(".")
        constraints = Gen.choicemap()
        for attr in attrs
            constraints[attr => :realization] = validation_df[i, attr]
        end

        (trace, weight) = Gen.generate(model, args, constraints)
        push!(score_list, weight)
        if length(k_score_list) < k
            push!(k_score_list, weight)
            push!(k_trace_list, trace)
            push!(k_n_s_list, validation_df[i, :negative_sample])
            continue
        end
        max_score = maximum(k_score_list)
        if weight < max_score
            max_i = findall(k_score_list .== max_score)[1]
            deleteat!(k_score_list, max_i)
            deleteat!(k_trace_list, max_i)
            deleteat!(k_n_s_list, max_i)
            push!(k_score_list, weight)
            push!(k_trace_list, trace)
            push!(k_n_s_list, validation_df[i, :negative_sample])
        end
        if i % floor(size(validation_df, 1) / 20) == 0.0
            print("-")
            println("$(round(i / size(validation_df, 1);digits=2)) done")
        end
    end
    precision = convert(Float32, sum(k_n_s_list)) / convert(Float32, k)
    recall = precision
    return (k_trace_list, k_score_list, k_n_s_list, recall, score_list)
end

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

function parse_ek_data(db::SQLite.DB, table::Symbol)
    patched_df = SQLite.Query(db, "SELECT catalog_id, article_id, destination, lower_bound, ek_amount, vk_amount, currency, unit, tax, set_id
                                   FROM $table") |> DataFrame
end

function k_most_improbable_neo(k, observation_df, validation_df, attrs, model, args)
    @assert typeof(observation_df) == DataFrame
    k_trace_list = []
    k_score_list = []
    for i in 1:size(validation_df, 1)
        constraints = Gen.choicemap()
        for attr in attrs
            constraints[attr => :realization] = validation_df[i, attr]
        end

        (trace, weight) = Gen.generate(model, args, constraints)
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
