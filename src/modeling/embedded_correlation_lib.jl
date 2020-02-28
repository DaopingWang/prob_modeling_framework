include("../custom_distributions/custom_distribution_lib.jl")
include("util.jl")
using Gen
using DataFrames: DataFrame, Missing
using LinearAlgebra, Statistics
using Distances

global NEAREST_SEPARATE_EMB_BUFFER = Dict()

function get_k_nearest_concat_emb(emb, emb_matrix, skip_indices, k)
    @assert k > 0
    k_emb_list = []
    k_dist_list = []
    k_index_list = []
    for (i, e) in enumerate(emb_matrix)
        if i in skip_indices
            continue
        end
        dist = Distances.cosine_dist(emb, e)
        if length(k_dist_list) < k
            push!(k_dist_list, dist)
            push!(k_emb_list, e)
            push!(k_index_list, i)
            continue
        end
        max_dist = maximum(k_dist_list)
        if dist < max_dist
            max_i = findall(k_dist_list .== max_dist)[1]
            deleteat!(k_dist_list, max_i)
            deleteat!(k_emb_list, max_i)
            deleteat!(k_index_list, max_i)
            push!(k_dist_list, dist)
            push!(k_emb_list, e)
            push!(k_index_list, i)
        end
    end
    (k_emb_list, k_dist_list, k_index_list)
end

function get_k_nearest_separate_emb(emb_vec, emb_matrix, skip_indices, k::Int)
    @assert k > 0
    global NEAREST_SEPARATE_EMB_BUFFER
    identifier = vcat(reduce(vcat, emb_vec), skip_indices)
    if haskey(NEAREST_SEPARATE_EMB_BUFFER, identifier)
        #@warn "using buffered emb_vec"
        return NEAREST_SEPARATE_EMB_BUFFER[identifier]
    else
    k_dist_index_df = DataFrame()
    k_dist_list = [[] for _=1:Threads.nthreads()]
    k_index_list = [[] for _=1:Threads.nthreads()]
    Threads.@threads for i = 1:size(emb_matrix, 1)
        if i in skip_indices
            continue
        end
        sum_dist = 0.
        for (j, v) in enumerate(emb_vec)
            sum_dist += abs(Distances.cosine_dist(v, emb_matrix[i, j]))
        end
        push!(k_dist_list[Threads.threadid()], sum_dist)
        push!(k_index_list[Threads.threadid()], i)
    end
    k_dist_index_df.dist = vcat(k_dist_list...)
    k_dist_index_df.index = vcat(k_index_list...)
    sort!(k_dist_index_df, [:dist], rev=false)
    if length(k_dist_index_df.index) < k
        @warn "only $(length(k_dist_index_df.index)) neighbors found, $k requested"
        NEAREST_SEPARATE_EMB_BUFFER[identifier] = k_dist_index_df.index
        return k_dist_index_df.index
    end
    NEAREST_SEPARATE_EMB_BUFFER[identifier] = k_dist_index_df.index[1:k]
    return k_dist_index_df.index[1:k]
    end
end

# O(nlogn)
@gen function embedding_co_occurrence(df::DataFrame,
                                    emb_df::DataFrame,
                                    emb_dict::Dict,
                                    attr_left::Array,
                                    attr_left_type::Array,
                                    attr_right::Symbol,
                                    attr_right_type::String,
                                    val_left::Array,
                                    k::Int)
    @assert length(attr_left) == length(attr_left_type) == length(val_left)
    # get data entries with same values as val_left
    cat_mask = [true for _=1:size(df, 1)]
    num_bucket_mask = [true for _=1:size(df, 1)]
    hard_cat_mask = [true for _=1:size(df, 1)]
    # time complexity n_col * n_observation * t_df_lookup = n_obs as n_col << n_obs
    #Threads.@threads 
    for i = 1:size(attr_left, 1)
        attr = attr_left[i]
        if attr_left_type[i] == "embedding"
            cat_mask = cat_mask .& get_categorical_mask(df, attr, [val_left[i]])
        elseif attr_left_type[i] == "numerical"
            # bucketize with val_left[i][2]
            num_bucket_mask = num_bucket_mask .& get_numerical_mask(df, attr, val_left[i][1], val_left[i][2])
        elseif attr_left_type[i] == "categorical"
            # non embedded cat attrs act like hard bound
            hard_cat_mask = hard_cat_mask .& get_categorical_mask(df, attr, [val_left[i]])
        else
            @error "unknown attr left type $(attr_left_type[i])"
        end
    end
    mask = cat_mask .& num_bucket_mask
    # n
    skip_indices = findall(mask .== true)
    # all entries with num lhs attr value that is outside bucket will be skipped when looking for neighbors
    # n
    append!(skip_indices, findall(num_bucket_mask .== false))
    append!(skip_indices, findall(hard_cat_mask .== false))

    # check how many observations are found. if less than k, add in neighbors
    num_hit = sum(mask)
    #if num_hit == 0
    #    @warn "Combination/single value not observed for $val_left"
    #end
    # n_obs log n_obs
    if num_hit < k
        @info "num_hit = $num_hit for $val_left, looking for neighbors"
        # combi/value rarely observed, need neighbors
        num_neighbor = k - num_hit

        emb_vec = []
        emb_attrs = []
        for (i, attr) in enumerate(attr_left)
            if attr_left_type[i] == "embedding"
                push!(emb_attrs, attr)
                push!(emb_vec, emb_dict[val_left[i]])
            end
        end

        emb_matrix = convert(Matrix, emb_df[!, emb_attrs])
        neighbor_index = get_k_nearest_separate_emb(emb_vec,
                                                    emb_matrix,
                                                    unique(skip_indices),
                                                    num_neighbor)

        for i in neighbor_index
            mask[i] = true
            info_list = convert(Array, df[i, :])
            @info "nearest neighbors: $info_list"
        end
    end
    # check attr right type, if categorical: local frequency info (total oc = k)
    # if numerical: normal with mean and std of k
    if attr_right_type == "categorical"
        associated_attr_right = df[mask, attr_right]
        associated_attr_right_unique = unique(associated_attr_right)
        associated_attr_right_prob = [sum(associated_attr_right .== a) for a in associated_attr_right_unique]

        # TODO NEW FOR INFERENCE, need better low score for non associated right values
        other_attr_right_unique = filter(x->x ∉ associated_attr_right_unique, unique(df[:, attr_right]))
        low_score = sum(associated_attr_right_prob) * 0.1 / size(other_attr_right_unique, 1)
        other_attr_right_prob = [low_score for _=1:size(other_attr_right_unique, 1)]

        probs = LinearAlgebra.normalize(vcat(associated_attr_right_prob, other_attr_right_prob), 1)
        return @trace(categorical_named(vcat(associated_attr_right_unique, other_attr_right_unique), probs), :realization)
    elseif attr_right_type == "numerical"
        m = sum(mask) <= 1 ? mean(df[:, attr_right]) : mean(df[mask, attr_right])
        s = sum(mask) <= 1 ? std(df[:, attr_right]) : std(df[mask, attr_right])
        return @trace(normal(m, s), :realization)
    else
        @error "unknown attr right type $attr_right_type"
    end
end
