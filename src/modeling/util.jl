include("../custom_distributions/custom_distribution_lib.jl")
using Gen
using SQLite
using DataFrames: DataFrame, rename!, DataFrameRow
using Statistics, LinearAlgebra
using CSV
using Logging

function make_constraints(df_row::DataFrameRow,
                        ignore_attrs)
    constraints = Gen.choicemap()
    for attr in names(df_row)
        if attr in ignore_attrs
            continue
        end
        constraints[attr => :realization] = df_row[attr]
    end
    return constraints
end

function read_tsv(meta_path, vec_path)
    #print(".")
    meta = CSV.read(meta_path; header=false) |> Matrix{}
    vec = CSV.read(vec_path; header=false, delim="\t") |> Matrix{Float64}
    vec_array = [vec[i, :] for i=1:size(vec, 1)]
    Dict(zip(meta, vec_array))
end

function get_categorical_mask(df, cat_attr, val)
    [df[i, cat_attr] in val for i=1:size(df, 1)]
end

function get_numerical_mask(df, num_attr, val, range)
    [abs(df[j, num_attr] - val) <= range for j=1:size(df, 1)]
end

function replace_with_emb(df, cat_attrs, num_attrs, emb_dict)
    emb_df = DataFrame()
    for attr in cat_attrs
        emb_df[!, attr] = [emb_dict[df[i, attr]] for i=1:size(df, 1)]
    end
    for attr in num_attrs
        emb_df[!, attr] = df[!, attr]
    end
    emb_df
end

function get_pr_roc(sorted_df, n_negative_sample)
    pr_df = DataFrame()
    roc_df = DataFrame()
    precision_list = []
    recall_list = []
    fpr_list = []

    n_positive_sample = size(sorted_df, 1) - n_negative_sample
    tp = 0.
    fp = 0.

    for ns in sorted_df.ns
        if ns == 1
            tp += 1.
        else
            fp += 1.
        end
        fn = n_negative_sample - tp
        tn = n_positive_sample - fp
        append!(precision_list, tp/(tp+fp))
        append!(recall_list, tp/(tp+fn))
        append!(fpr_list, fp/(fp+tn))
    end
    # precision == ppv
    pr_df.precision = precision_list
    pr_df.recall = recall_list

    # recall == tpr
    roc_df.recall = recall_list
    roc_df.fpr = fpr_list
    return pr_df, roc_df
end

function k_most_improbable(k::Int, df::DataFrame, attrs::Array{Symbol, 1}, model, args)
    sorted_df = deepcopy(df)
    #score_list = [[] for _=1:Threads.nthreads()]
    #ns_list = [[] for _=1:Threads.nthreads()]
    score_list = []
    #ns_list = []

    # SOME OPERATION HERE IS NOT THREAD-SAFE AND BREAKS EVERYTHING...
    #Threads.@threads
    for i = 1:size(df, 1)
        # progress bar
        if i % floor(size(df, 1) / 20) == 0.0
            print("k_most_improbable(): ")
            println("$(round(i / size(df, 1);digits=2)) done")
        end

        constraints = Gen.choicemap()
        for attr in attrs
            constraints[attr => :realization] = df[i, attr]
        end

        (trace, weight) = Gen.generate(model, args, constraints)
        #push!(score_list[Threads.threadid()], weight)
        #push!(ns_list[Threads.threadid()], df[i, :negative_sample])

        push!(score_list, weight)
        #push!(ns_list, df[i, :negative_sample])
    end
    #sorted_df.score = vcat(score_list...)
    #sorted_df.ns = vcat(ns_list...)
    sorted_df.score = score_list
    #sorted_df.ns = ns_list

    sort!(sorted_df, [:score], rev=false)
    retdf = sorted_df[1:k, :]
    println("---------------")
    return sorted_df, retdf.negative_sample
end

function neo_negative_sampler(df::DataFrame, pivot::Symbol, attrs::Array, n::Int)
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
            elseif bernoulli(0.2)
                sampled = true
            else
                # dont neg samp this attr
                continue
            end
            pos_attr_vals = unique(df[df[:, pivot].==n_s_buffer[pivot], attr])
            neg_attr_vals = filter(x->x ∉ pos_attr_vals, attrs_values_unique[j])
            n_s_buffer[attr] = uniform_categorical(neg_attr_vals)
        end
        push!(n_s_df_rows, deepcopy(n_s_buffer))
    end
    return n_s_df_rows
end

function get_trace_vals(trace, attrs)
    retval = []
    for attr in attrs
        append!(retval, [trace[attr => :realization]])
    end
    return retval
end

function get_k_most_attr_val(attr::Symbol, df::DataFrame, k::Int)
    k_attr_val_list = []
    k_attr_count_list = []
    for val in unique(df[:, attr])
        count = sum(df[:, attr] .== val)
        if size(k_attr_val_list, 1) < k
            push!(k_attr_val_list, val)
            push!(k_attr_count_list, count)
            continue
        end
        min_count = minimum(k_attr_count_list)
        if count > min_count
            min_i = findall(k_attr_count_list .== min_count)[1]
            deleteat!(k_attr_count_list, min_i)
            deleteat!(k_attr_val_list, min_i)
            push!(k_attr_count_list, count)
            push!(k_attr_val_list, val)
        end
    end
    return k_attr_val_list, k_attr_count_list
end
