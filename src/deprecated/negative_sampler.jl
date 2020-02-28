cd(dirname(@__FILE__))
include("custom_distributions/custom_distribution_lib.jl")
using Gen
using DataFrames
using Logging, Statistics

function negative_sampler(df, attr, ignore_attrs, n_s_per_unique_attr)
    attr_values_unique = unique(df[:, attr])
    other_attrs = filter(x->x!=attr, names(df))
    other_attrs = filter(x->x ∉ ignore_attrs, other_attrs)
    n_s_df_rows = []
    # for each city
    for (i, attr_val) in enumerate(attr_values_unique)
        # get all entries with current city val
        attr_val_df = df[df[:, attr].==attr_val, :]
        # samples 5 negative samples for this city
        for _ = 1:n_s_per_unique_attr
            n_s_buffer = DataFrameRow(attr_val_df, 1)
            # choose one attr to make dirty, e.g. state
            neg_attr = uniform_categorical(other_attrs)
            # which states occurred together with city?
            pos_attr_vals = unique(attr_val_df[:, neg_attr])
            # which have not?
            neg_attr_vals = unique(filter(x->x ∉ pos_attr_vals, df[:, neg_attr]))
            # pick one state that has not occurred together with city
            n_s_buffer[neg_attr] = uniform_categorical(neg_attr_vals)
            # for other attrs beside city and state, choose randomly whether they
            # are also to be changed
            for other_other_attr in filter(x->x!=neg_attr, other_attrs)
                pos_other_other_attr_vals = unique(attr_val_df[:, other_other_attr])
                neg_other_other_attr_vals = unique(filter(x->x ∉ pos_other_other_attr_vals, df[:, other_other_attr]))
                if bernoulli(0.1)
                    # add wrong val
                    n_s_buffer[other_other_attr] = uniform_categorical(neg_other_other_attr_vals)
                else
                    # add occurred val
                    n_s_buffer[other_other_attr] = uniform_categorical(pos_other_other_attr_vals)
                end
            end
            for ignore_attr in ignore_attrs
                pos_ignore_attr_vals = unique(attr_val_df[:, ignore_attr])
                n_s_buffer[ignore_attr] = uniform_categorical(pos_ignore_attr_vals)
            end
            push!(n_s_df_rows, deepcopy(n_s_buffer))
        end
    end
    return n_s_df_rows
end
