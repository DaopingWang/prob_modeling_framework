include("../custom_distributions/custom_distribution_lib.jl")
include("../modeling/util.jl")
using Gen
using DataFrames: DataFrame, Missing, DataFrameRow
using LinearAlgebra, Statistics, Distances

global NEIGHBORHOOD_BUFFER = []
global PROBABILITY_BUFFER = []
global INFER_ATTR_BUFFER = nothing
global INFER_ATTR_EMB_BUFFER = []

function get_k_neighbors(attr_emb::Array{Float64,1},
                        attr_embs::Array{Array{Float64,1},1},
                        emb_cat_dict::Dict,
                        k::Int)
    @assert k > 0
    k_neighbor_list = []
    k_dist_list = []
    for neighbor_emb in attr_embs
        dist = abs(Distances.cosine_dist(attr_emb, neighbor_emb))
        if length(k_dist_list) < k
            push!(k_dist_list, dist)
            push!(k_neighbor_list, emb_cat_dict[neighbor_emb])
            continue
        end
        max_dist = maximum(k_dist_list)
        if dist < max_dist
            max_i = findall(k_dist_list .== max_dist)[1]
            deleteat!(k_dist_list, max_i)
            deleteat!(k_neighbor_list, max_i)
            push!(k_dist_list, dist)
            push!(k_neighbor_list, emb_cat_dict[neighbor_emb])
        end
    end
    return k_neighbor_list
end

@gen function neighborhood_proposal(infer_attr::Symbol,
                                infer_attr_emb::Array{Float64,1},
                                infer_attr_embs::Array{Array{Float64,1},1},
                                emb_cat_dict::Dict,
                                neighborhood_size::Int)
    global NEIGHBORHOOD_BUFFER
    global PROBABILITY_BUFFER
    global INFER_ATTR_BUFFER
    global INFER_ATTR_EMB_BUFFER
    if (INFER_ATTR_BUFFER != infer_attr) || (INFER_ATTR_EMB_BUFFER != infer_attr_emb)
        #@warn "Initialize buffer"
        unique_infer_attr_embs = unique(infer_attr_embs)
        NEIGHBORHOOD_BUFFER = get_k_neighbors(infer_attr_emb,
                                            filter(x->x!=infer_attr_emb, unique_infer_attr_embs),
                                            emb_cat_dict,
                                            neighborhood_size)
        NEIGHBORHOOD_BUFFER = vcat(NEIGHBORHOOD_BUFFER,
                                        filter(x->x ∉ NEIGHBORHOOD_BUFFER, [emb_cat_dict[e] for e in unique_infer_attr_embs]))
        score_high = 5.
        PROBABILITY_BUFFER = [score_high for _=1:neighborhood_size]
        PROBABILITY_BUFFER = vcat(PROBABILITY_BUFFER,
                                        ones(size(unique_infer_attr_embs, 1) - neighborhood_size))
        PROBABILITY_BUFFER = LinearAlgebra.normalize(PROBABILITY_BUFFER, 1)
        INFER_ATTR_BUFFER = infer_attr
        INFER_ATTR_EMB_BUFFER = infer_attr_emb
    end

    @trace(categorical_named(NEIGHBORHOOD_BUFFER, PROBABILITY_BUFFER), infer_attr => :realization)
end

function do_neighborhood_inference(model::GenerativeFunction,
                                model_args::Tuple,
                                infer_attr::Symbol,
                                emb_df::DataFrame,
                                cat_emb_dict::Dict,
                                emb_cat_dict::Dict,
                                observation::DataFrameRow,
                                neighborhood_size::Int,
                                amount_of_computation::Int)
    constraints = make_constraints(observation, [infer_attr])
    infer_attr_emb = cat_emb_dict[observation[infer_attr]]
    infer_attr_embs = emb_df[infer_attr]
    # invoke the variant of importance_resampling that accepts a custom proposal (dest_proposal)
    # the arguments to the custom proposal are (measurements, scene)
    (trace, _) = Gen.importance_resampling(model,
                                        model_args,
                                        constraints,
                                        neighborhood_proposal,
                                        (infer_attr, infer_attr_emb, infer_attr_embs, emb_cat_dict, neighborhood_size),
                                        amount_of_computation)

    return trace
end
