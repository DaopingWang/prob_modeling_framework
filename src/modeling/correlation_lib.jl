include("../custom_distributions/custom_distribution_lib.jl")
include("util.jl")
using Gen
using DataFrames: DataFrame, Missing
using LinearAlgebra, Statistics


##############################################################################
@gen function categorical_co_occurrence(df::DataFrame,
                                    attr_left::Array{Symbol, 1},
                                    attr_left_type::Array,
                                    attr_right::Symbol,
                                    val_left::Array,
                                    strict_dist::Bool)
    @assert length(attr_left) == length(attr_left_type) == length(val_left)

    mask = [true for _=1:size(df, 1)]
    for (i, attr) in enumerate(attr_left)
        buffer = [true for _=1:size(df, 1)]
        if attr_left_type[i] == "categorical"
            buffer = mask .& get_categorical_mask(df, attr, [val_left[i]])
        else
            buffer = mask .& get_numerical_mask(df, attr, val_left[i][1], val_left[i][2])
        end

        # combination not found in observation, dont update mask
        if (!strict_dist) && (sum(buffer) == 0)
            @warn "cat combi failed: $attr"
            continue
        end
        mask = buffer
    end

    associated_attr_right = df[mask, attr_right]
    associated_attr_right_unique = unique(associated_attr_right)
    associated_attr_right_prob = [sum(associated_attr_right .== a) for a in associated_attr_right_unique]
    #probs = LinearAlgebra.normalize(associated_attr_right_prob, 1)
    #return @trace(categorical_named(associated_attr_right_unique, probs), :realization)

    other_attr_right_unique = filter(x->x ∉ associated_attr_right_unique, unique(df[:, attr_right]))
    low_score = sum(associated_attr_right_prob) * 0.1 / size(other_attr_right_unique, 1)
    other_attr_right_prob = [low_score for _=1:size(other_attr_right_unique, 1)]

    probs = LinearAlgebra.normalize(vcat(associated_attr_right_prob, other_attr_right_prob), 1)
    return @trace(categorical_named(vcat(associated_attr_right_unique, other_attr_right_unique), probs), :realization)
end

# O(n)
@gen function numerical_co_occurrence(df::DataFrame,
                                    attr_left::Array,
                                    attr_left_type::Array,
                                    attr_right::Symbol,
                                    val_left::Array,
                                    greater_zero::Bool,
                                    strict_dist::Bool)
    @assert length(attr_left) == length(attr_left_type) == length(val_left)
    use_this_normal = greater_zero ? half_normal : normal

    # all observed attr_2 values that co-occurred with val_1, with duplicates
    mask = [true for _=1:size(df, 1)]
    for (i, attr) in enumerate(attr_left)
        buffer = [true for _=1:size(df, 1)]
        if attr_left_type[i] == "categorical"
            buffer = mask .& get_categorical_mask(df, attr, [val_left[i]])
        else
            buffer = mask .& get_numerical_mask(df, attr, val_left[i][1], val_left[i][2])
        end

        # combination not found in observation, dont update mask
        if (!strict_dist) && (sum(buffer) == 0)
            @warn "num combi failed: $attr"
            continue
        end
        mask = buffer
    end
    m = (sum(mask) <= 1) ? mean(df[:, attr_right]) : mean(df[mask, attr_right])
    s = (sum(mask) <= 1) ? std(df[:, attr_right]) : std(df[mask, attr_right])
    realization =  @trace(use_this_normal(m, s), :realization)
end
