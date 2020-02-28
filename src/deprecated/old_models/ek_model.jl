include("custom_distributions/custom_distribution_lib.jl")

using SQLite
using Gen
using CSV
using DataFrames: DataFrame, names
using LinearAlgebra
using Statistics
using Logging
using Random

ATTRIBUTES = [:catalog_id,
            :article_id,
            :destination,
            :lower_bound,
            :ek_amount,
            :vk_amount,
            :currency,
            :unit,
            :tax,
            :set_id]

function parse_data(db::SQLite.DB, table::Symbol)
    patched_df = SQLite.Query(db, "SELECT catalog_id, article_id, destination, lower_bound, ek_amount, vk_amount, currency, unit, tax, set_id
                                   FROM $table") |> DataFrame
end

function make_constraints(df_row)
    constraints = Gen.choicemap()
    for (i, attr) in enumerate(names(df_row))
        if attr in ATTRIBUTES
            constraints[attr => :realization] = attr == :ek_amount ? df_row[i] : Symbol(df_row[i])
        end
    end
    return constraints
end

@gen function uniformly_categorical(df, attr)
    @assert typeof(df) == DataFrame
    @assert typeof(attr) == Symbol

    attr_unique_rows = unique(df[:, attr])
    attr_probs = ones(length(attr_unique_rows))
    realization = @trace(named_categorical(normalize(attr_probs, 1), map(Symbol, attr_unique_rows)), :realization)
end

@gen function statistics(df, attr)
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
    realization = @trace(named_categorical(normalize(probs, 1), map(Symbol,attr_unique_rows)), :realization)
end

# Special FD-Case of multiple lefthand side arguments
# e.g. A=a, B=b, C=c -> Y=y
@gen function functional_dependency(df,
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
    realization = @trace(named_categorical(normalize(attr_right_probs, 1), map(Symbol, attr_right_unique_rows)), :realization)
end

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
    local_attr_right_std = isnan(local_attr_right_std) ? 0.01 : local_attr_right_std

    if care_global_oc || isnan(local_attr_right_mean)
        global_attr_right_mean = mean(collect(skipmissing(df[:, attr_right])))
        global_attr_right_std = std(collect(skipmissing(df[:, attr_right])))
        global_attr_right_std = isnan(global_attr_right_std) ? 0.01 : global_attr_right_std
        if isnan(local_attr_right_mean)
            return @trace(use_this_normal(global_attr_right_mean, global_attr_right_std), :realization)
        end
        mean_var = mean([local_attr_right_mean, global_attr_right_mean])
        std_var = mean([local_attr_right_std, global_attr_right_std])
        return @trace(use_this_normal(mean_var, std_var), :realization)
    end
    realization = @trace(use_this_normal(local_attr_right_mean, local_attr_right_std), :realization)
end

@gen function ek_model(realization_df)
    @info "-----------------ADV"
    # 1. Pick catalog_id
    # catalog_id = @trace(statistics(realization_df, :catalog_id), :catalog_id)
    catalog_id = @trace(uniformly_categorical(realization_df, :catalog_id), :catalog_id)
    @info "CID: $catalog_id"
    # 2. Pick article_id corresponding to observed occurrence. If no correlation
    # between observed occurrence: article_id should be drawn from a uniform discrete
    # distribution.
    article_id = @trace(functional_dependency(realization_df,
                                            [:catalog_id,],
                                            :article_id,
                                            [String(catalog_id),],
                                            false), :article_id)
    @info "AID: $article_id"

    # 3. Pick destination: We assume that the same product from the same supplier
    # comes from the same country. (WHICH COULD VERY LIKELY BE WRONG!)
    destination = @trace(functional_dependency(realization_df,
                                            [:catalog_id, :article_id],
                                            :destination,
                                            [String(catalog_id), String(article_id)],
                                            false), :destination)
    @info "Destination: $destination"

    # 4. Pick lower bound
    # Chance to pick a lower bound that has been observed together with aid is high.
    # For the same article from the same supplier, higher lower bound should imply
    # lower ek. => Vertical learning
    lower_bound = @trace(functional_dependency(realization_df,
                                            [:catalog_id, :article_id],
                                            :lower_bound,
                                            [String(catalog_id), String(article_id)],
                                            false), :lower_bound)
    @info "LB: $lower_bound"

    # 5. Pick currency
    currency = @trace(functional_dependency(realization_df,
                                            [:destination],
                                            :currency,
                                            [String(destination)],
                                            false), :currency)
    @info "Currency: $currency"

    # 6. Pick unit
    unit = @trace(functional_dependency(realization_df,
                                        [:article_id],
                                        :unit,
                                        [String(article_id)],
                                        false), :unit)
    @info "Unit: $unit"

    # 7. Pick tax
    tax = @trace(functional_dependency(realization_df,
                                    [:article_id, :destination],
                                    :tax,
                                    [String(article_id), String(destination)],
                                    false), :tax)
    @info "Tax: $tax"

    # Pick set_id
    set_id = @trace(functional_dependency(realization_df,
                                        [:catalog_id, :article_id],
                                        :set_id,
                                        [String(catalog_id), String(article_id)],
                                        false), :set_id)
    @info "SID: $set_id"

    # 9. Pick ek
    # ek is heavily correlated with other attributes. Here we use multiple FDs
    # to make ek suggestions and sample final ek, using mean and std of the
    # suggestions
    # TODO replace these FDs with embeddings
    # From article_id
    aid_unit_ek = numerical_functional_dependency(realization_df,
                                            [:article_id, :unit],
                                            :ek_amount,
                                            [String(article_id), String(unit)],
                                            true,
                                            false)

    aid_cid_unit_ek = numerical_functional_dependency(realization_df,
                                            [:article_id, :catalog_id, :unit],
                                            :ek_amount,
                                            [String(article_id), String(catalog_id), String(unit)],
                                            true,
                                            false)
    set_id_unit_ek = numerical_functional_dependency(realization_df,
                                                    [:set_id, :unit],
                                                    :ek_amount,
                                                    [String(set_id), String(unit)],
                                                    true,
                                                    false)
    mean_ek = mean([aid_unit_ek, aid_cid_unit_ek, set_id_unit_ek])
    std_ek = max(std([aid_cid_unit_ek, aid_unit_ek, set_id_unit_ek]), 0.0001)
    ek_amount = @trace(half_normal(mean_ek, std_ek), :ek_amount => :realization)
    @info "EK: $ek_amount"

end

db = SQLite.DB()
patched_table = CSV.File(joinpath(@__DIR__, "../data/mercateo/patched.csv")) |> SQLite.load!(db, "patched_table")


patched_df = parse_data(db, :patched_table)

# traces = [Gen.simulate(ek_model, (patched_df,)) for _=1:2]

# Evaluate marginal probabilities of data entries
#=for i = 1:5
    constraints = make_constraints(patched_df[i, :])
    (trace, weight) = Gen.generate(ek_model, (patched_df,), constraints)
    println("Loglikelihood: $weight")
end=#
