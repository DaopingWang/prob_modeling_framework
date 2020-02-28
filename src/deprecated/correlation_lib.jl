include("../custom_distributions/custom_distribution_lib.jl")
include("util.jl")
using Gen
using DataFrames: DataFrame, Missing
using LinearAlgebra, Statistics

@gen function uniformly_categorical(df, attr)
    @assert typeof(df) == DataFrame
    @assert typeof(attr) == Symbol

    attr_unique_rows = unique(df[:, attr])
    attr_probs = ones(length(attr_unique_rows))
    realization = @trace(named_categorical(LinearAlgebra.normalize(attr_probs, 1), map(Symbol, attr_unique_rows)), :realization)
end

@gen function occurrence(df, attr)
    @assert typeof(df) == DataFrame
    @assert typeof(attr) == Symbol

    attr_dup_rows = df[:, attr]
    attr_unique_rows = unique(attr_dup_rows)
    attr_oc = [sum(attr_dup_rows .== val) for val in attr_unique_rows]
    total_attr_oc = sum(attr_oc)

    probs = ones(length(attr_unique_rows))
    for (i, val) in enumerate(attr_unique_rows)
        prob = beta(attr_oc[i], total_attr_oc - attr_oc[i])
        probs[i] *= prob
    end
    realization = @trace(named_categorical(LinearAlgebra.normalize(probs, 1), map(Symbol,attr_unique_rows)), :realization)
end

@gen function functional_dependency(df, attr_1, attr_2, val_1, care_global_oc)
    @assert typeof(df) == DataFrame
    @assert typeof(attr_1) == typeof(attr_2) == Symbol
    @assert typeof(care_global_oc) == Bool
    # all observed values, with duplicates
    attr_1_dup_rows = df[:, attr_1]
    attr_2_dup_rows = df[:, attr_2]
    # unique attr_2 values
    attr_2_unique_rows = unique(attr_2_dup_rows)
    # all observed attr_2 values that co-occurred with val_1, with duplicates
    associated_attr_2 = df[attr_1_dup_rows .== val_1, attr_2]

    if care_global_oc
        global_attr_2_oc = [sum(attr_2_dup_rows .== val) for val in attr_2_unique_rows]
        total_global_attr_2_oc = sum(global_attr_2_oc)
    end
    local_attr_2_oc = [if val in associated_attr_2
                          sum(associated_attr_2 .== val)
                       else
                          1
                       end for val in attr_2_unique_rows]
    total_local_attr_2_oc = sum(local_attr_2_oc)

    attr_2_probs = ones(length(attr_2_unique_rows))
    for (i, val) in enumerate(attr_2_unique_rows)
        local_val_prob = beta(local_attr_2_oc[i], total_local_attr_2_oc - local_attr_2_oc[i])
        attr_2_probs[i] *= local_val_prob
        if care_global_oc
            global_val_prob = beta(global_attr_2_oc[i], total_global_attr_2_oc - global_attr_2_oc[i])
            attr_2_probs[i] *= global_val_prob
        end
    end

    #attr_2_probs = softmax(local_attr_2_oc)
    realization = @trace(named_categorical(LinearAlgebra.normalize(attr_2_probs, 1), map(Symbol, attr_2_unique_rows)), :realization)
end

# Special FD-Case of multiple lefthand side arguments
# e.g. A=a, B=b, C=c -> Y=y
# TODO this fd function is useful when assuming Pr(AR|AL=al)>0 for all ar in dom(AR)
# Though the way it assigns probability to ar is not necessarily meaningful (high score
# low score stuff)
@gen function co_occurrence(df,
                            attr_left,
                            attr_right,
                            val_left,
                            care_global_oc)
    @assert typeof(attr_left) == Array{Symbol, 1}
    @assert typeof(attr_right) == Symbol
    @assert typeof(care_global_oc) == Bool

    attr_left_dup_rows = [collect(skipmissing(df[:, attr])) for attr in attr_left]
    attr_right_dup_rows = collect(skipmissing(df[:, attr_right]))
    attr_right_unique_rows = unique(attr_right_dup_rows)
    # all observed attr_2 values that co-occurred with val_1, with duplicates
    mask = [true for _=1:size(df, 1)]
    for (i, val) in enumerate(attr_left)
        mask = mask .& (attr_left_dup_rows[i] .== val_left[i])
    end
    associated_attr_right = df[mask, attr_right]
    if care_global_oc
        global_attr_right_oc = [sum(attr_right_dup_rows .== val) for val in attr_right_unique_rows]
        total_global_attr_right_oc = sum(global_attr_right_oc)
    end
    # TODO We have a lot of data in contrast to the low number of occurrence of each attr. val
    # Think about how to reward occurred value and how to punish others without
    # stating an hard constraint
    low_score = log(length(attr_right_unique_rows))
    high_score = low_score * length(attr_right_unique_rows) * 10
    local_attr_right_oc = [if val in associated_attr_right
                              # sum(associated_attr_right .== val) * 100
                              high_score
                           else
                              low_score
                           end for val in attr_right_unique_rows]
    total_local_attr_right_oc = sum(local_attr_right_oc)
    attr_right_probs = ones(length(attr_right_unique_rows))
    for (i, val) in enumerate(attr_right_unique_rows)
        local_val_prob = beta(local_attr_right_oc[i], total_local_attr_right_oc - local_attr_right_oc[i])
        attr_right_probs[i] *= local_val_prob
        if care_global_oc
            global_val_prob = beta(global_attr_right_oc[i], total_global_attr_right_oc - global_attr_right_oc[i])
            attr_right_probs[i] *= global_val_prob
        end
    end
    realization = @trace(named_categorical(LinearAlgebra.normalize(attr_right_probs, 1), map(Symbol, attr_right_unique_rows)), :realization)
end

# Given lefthand side attribute values, sample numerical value for one righthand
# side attribute
@gen function numerical_functional_dependency(df,
                                            attr_left,
                                            attr_right,
                                            val_left,
                                            greater_zero,
                                            care_global_oc)
    @assert typeof(attr_left) == Array{Symbol, 1}
    @assert typeof(attr_right) == Symbol
    @assert typeof(care_global_oc) == typeof(greater_zero) == Bool

    # TODO implement pd for positive normal distributed vars
    use_this_normal = greater_zero ? half_normal : normal
    #use_this_normal = normal
    # all lefthand side attribute values w/o missings
    attr_left_dup_rows = [collect(skipmissing(df[:, attr])) for attr in attr_left]

    # all observed attr_2 values that co-occurred with val_1, with duplicates
    mask = [true for _=1:size(df, 1)]
    for (i, val) in enumerate(attr_left)
        mask = mask .& (attr_left_dup_rows[i] .== val_left[i])
    end
    associated_attr_right = df[mask, attr_right]

    # mean and std of righthand side values that occurred with given left values
    local_attr_right_mean = mean(collect(skipmissing(associated_attr_right)))
    local_attr_right_std = std(collect(skipmissing(associated_attr_right)))
    local_attr_right_std = isnan(local_attr_right_std) ? 1 : local_attr_right_std

    if care_global_oc || isnan(local_attr_right_mean)
        global_attr_right_mean = mean(collect(skipmissing(df[:, attr_right])))
        global_attr_right_std = std(collect(skipmissing(df[:, attr_right])))
        global_attr_right_std = isnan(global_attr_right_std) ? 1 : global_attr_right_std
        if isnan(local_attr_right_mean)
            return @trace(use_this_normal(global_attr_right_mean, global_attr_right_std), :realization)
        end
        mean_var = mean([local_attr_right_mean, global_attr_right_mean])
        std_var = mean([local_attr_right_std, global_attr_right_std])
        return @trace(use_this_normal(mean_var, std_var), :realization)
    end
    realization = @trace(use_this_normal(local_attr_right_mean, local_attr_right_std), :realization)
end
