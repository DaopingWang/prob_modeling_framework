cd(dirname(@__FILE__))
include("modeling/embedded_correlation_lib.jl") # backend generation functions
include("modeling/correlation_lib.jl") # backend generation functions
include("modeling/util.jl")

using SQLite
using CSV
using DataFrames: DataFrame, Missing, nonunique
using Statistics, LinearAlgebra, Logging, Random
using Gen

# generates PDB model for Mercateo article dataset, no embeddings and no ICs involved
@gen function ek_plain_No_Rules_model(df::DataFrame)
    catalog_ids = unique(df.catalog_id)
    catalog_id = @trace(uniform_categorical(catalog_ids), :catalog_id => :realization)

    keywords_ = unique(df.keywords)
    keywords = @trace(uniform_categorical(keywords_), :keywords => :realization)

    article_ids = unique(df.article_id)
    article_id = @trace(uniform_categorical(article_ids), :article_id => :realization)

    #lower_bounds = unique(df.lower_bound)
    #lower_bound = @trace(uniform_categorical(lower_bounds), :lower_bound => :realization)

    units = unique(df.unit)
    unit = @trace(uniform_categorical(units), :unit => :realization)

    manufacturer_names = unique(df.manufacturer_name)
    manufacturer_name = @trace(uniform_categorical(manufacturer_names), :manufacturer_name => :realization)

    set_ids = unique(df.set_id)
    set_id = @trace(uniform_categorical(set_ids), :set_id => :realization)

    ek_amount = @trace(numerical_co_occurrence(df,
                                            [:catalog_id, :set_id, :keywords, :unit, :manufacturer_name],
                                            ["categorical", "categorical", "categorical", "categorical", "categorical"],
                                            :ek_amount,
                                            [catalog_id, set_id, keywords, unit, manufacturer_name],
                                            false,
                                            true), :ek_amount)
    #=vk_amount = @trace(numerical_co_occurrence(df,
                                            [:article_id, :set_id, :keywords, :unit, :manufacturer_name, :lower_bound],
                                            ["categorical", "categorical", "categorical", "categorical", "categorical", "categorical"],
                                            :vk_amount,
                                            [article_id, set_id, keywords, unit, manufacturer_name, lower_bound],
                                            false,
                                            true), :vk_amount)=#
end

# generates PDB model for Mercateo article dataset, no ICs involved
@gen function ek_emb_No_Rules_model(df::DataFrame, emb_df::DataFrame, emb_dict::Dict)
    catalog_ids = unique(df.catalog_id)
    catalog_id = @trace(uniform_categorical(catalog_ids), :catalog_id => :realization)

    keywords_ = unique(df.keywords)
    keywords = @trace(uniform_categorical(keywords_), :keywords => :realization)

    article_ids = unique(df.article_id)
    article_id = @trace(uniform_categorical(article_ids), :article_id => :realization)

    #lower_bounds = unique(df.lower_bound)
    #lower_bound = @trace(uniform_categorical(lower_bounds), :lower_bound => :realization)

    units = unique(df.unit)
    unit = @trace(uniform_categorical(units), :unit => :realization)

    manufacturer_names = unique(df.manufacturer_name)
    manufacturer_name = @trace(uniform_categorical(manufacturer_names), :manufacturer_name => :realization)

    set_ids = unique(df.set_id)
    set_id = @trace(uniform_categorical(set_ids), :set_id => :realization)

    ek_neighborhood_size = 10
    ek_amount = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:catalog_id, :set_id, :keywords, :unit, :manufacturer_name],
                                            ["embedding", "embedding", "embedding", "categorical", "embedding"],
                                            :ek_amount,
                                            "numerical",
                                            [catalog_id, set_id, keywords, unit, manufacturer_name],
                                            ek_neighborhood_size), :ek_amount)
#=
    vk_neighborhood_size = 10
    vk_amount = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:article_id, :set_id, :keywords, :unit, :manufacturer_name, :lower_bound],
                                            ["embedding", "embedding", "embedding", "embedding", "embedding", "embedding"],
                                            :vk_amount,
                                            "numerical",
                                            [article_id, set_id, keywords, unit, manufacturer_name, lower_bound],
                                            vk_neighborhood_size), :vk_amount)=#
end

# generates PDB model for Mercateo article dataset, no embeddings involved
@gen function ek_plain_model(df::DataFrame)
    catalog_ids = unique(df.catalog_id)
    catalog_id = @trace(uniform_categorical(catalog_ids), :catalog_id => :realization)

    keywords_ = unique(df.keywords)
    keywords = @trace(uniform_categorical(keywords_), :keywords => :realization)

    #=article_id = @trace(categorical_co_occurrence(df,
                                            [:catalog_id,],
                                            ["categorical"],
                                            :article_id,
                                            [catalog_id],
                                            false), :article_id)=#
    article_ids = unique(df.article_id)
    article_id = @trace(uniform_categorical(article_ids), :article_id => :realization)

#=    lower_bound = @trace(categorical_co_occurrence(df,
                                            [:article_id,],
                                            ["categorical"],
                                            :lower_bound,
                                            [article_id],
                                            false), :lower_bound)

    ean = @trace(categorical_co_occurrence(df,
                                            [:keywords,],
                                            ["categorical"],
                                            :ean,
                                            [keywords],
                                            true), :ean)=#

    unit = @trace(categorical_co_occurrence(df,
                                            [:keywords],
                                            ["categorical"],
                                            :unit,
                                            [keywords],
                                            false), :unit)

    manufacturer_name = @trace(categorical_co_occurrence(df,
                                            [:keywords,],
                                            ["categorical"],
                                            :manufacturer_name,
                                            [keywords],
                                            false), :manufacturer_name)

    set_id = @trace(categorical_co_occurrence(df,
                                            [:keywords, :manufacturer_name, :unit],
                                            ["categorical", "categorical", "categorical"],
                                            :set_id,
                                            [keywords, manufacturer_name, unit],
                                            false), :set_id)

    ek_amount = @trace(numerical_co_occurrence(df,
                                            [:catalog_id, :set_id, :keywords, :unit, :manufacturer_name],
                                            ["categorical", "categorical", "categorical", "categorical", "categorical"],
                                            :ek_amount,
                                            [catalog_id, set_id, keywords, unit, manufacturer_name],
                                            false,
                                            true), :ek_amount)
#=
    vk_amount = @trace(numerical_co_occurrence(df,
                                            [:ek_amount, :lower_bound, :keywords],
                                            ["categorical", "categorical", "categorical"],
                                            :vk_amount,
                                            [ek_amount, lower_bound, keywords],
                                            false,
                                            true), :vk_amount)=#
end

# generates PDB model for Mercateo article dataset
#= df has the form:
│ Row  │ article_id │ catalog_id │ unit    │ keywords                │ manufacturer_name       │ set_id         │ ek_amount │ negative_sample │
│      │ String⍰    │ String⍰    │ String⍰ │ Union{Missing, String}  │ Union{Missing, String}  │ String⍰        │ Float64⍰  │ Int64⍰          │
├──────┼────────────┼────────────┼─────────┼─────────────────────────┼─────────────────────────┼────────────────┼───────────┼─────────────────┤
│ 1    │ 1546192-BP │ 102        │ C62     │ Schlüsselschrank        │ Durable                 │ 102-1546192-BP │ 77.44     │ 0               │
│ 2    │ 602067     │ 103        │ C62     │ Schlüsselschrank        │ Durable                 │ 102-1546192-BP │ 86.25     │ 0               │
│ 3    │ 88300      │ 114        │ C62     │ Schlüsseltresor         │ Durable                 │ 102-1546192-BP │ 100.72    │ 0               │
=#
@gen function ek_emb_model(df::DataFrame, emb_df::DataFrame, emb_dict::Dict)
    catalog_ids = unique(df.catalog_id)
    catalog_id = @trace(uniform_categorical(catalog_ids), :catalog_id => :realization)

    keywords_ = unique(df.keywords)
    keywords = @trace(uniform_categorical(keywords_), :keywords => :realization)

    #=aid_neighborhood_size = 1
    article_id = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:catalog_id,],
                                            ["embedding"],
                                            :article_id,
                                            "categorical",
                                            [catalog_id],
                                            aid_neighborhood_size), :article_id)=#
    article_ids = unique(df.article_id)
    article_id = @trace(uniform_categorical(article_ids), :article_id => :realization)

#=    lb_neighborhood_size = 1
    lower_bound = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:article_id,],
                                            ["embedding"],
                                            :lower_bound,
                                            "categorical",
                                            [article_id],
                                            lb_neighborhood_size), :lower_bound)

    ean = @trace(categorical_co_occurrence(df,
                                            [:keywords,],
                                            ["categorical"],
                                            :ean,
                                            [keywords],
                                            true), :ean)=#

    unit_neighborhood_size = 5
    unit = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:keywords,],
                                            ["embedding"],
                                            :unit,
                                            "categorical",
                                            [keywords],
                                            unit_neighborhood_size), :unit)

    mn_neighborhood_size = 10
    manufacturer_name = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:keywords,],
                                            ["embedding"],
                                            :manufacturer_name,
                                            "categorical",
                                            [keywords],
                                            mn_neighborhood_size), :manufacturer_name)

    sid_neighborhood_size = 10
    set_id = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:keywords, :manufacturer_name, :unit],
                                            ["embedding", "embedding", "categorical"],
                                            :set_id,
                                            "categorical",
                                            [keywords, manufacturer_name, unit],
                                            sid_neighborhood_size), :set_id)
    ek_neighborhood_size = 10
    ek_amount = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:catalog_id, :set_id],
                                            ["embedding", "embedding"],
                                            :ek_amount,
                                            "numerical",
                                            [catalog_id, set_id],
                                            ek_neighborhood_size), :ek_amount)
    #=ek_amount = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:catalog_id, :set_id, :keywords, :unit, :manufacturer_name],
                                            ["embedding", "embedding", "embedding", "categorical", "embedding"],
                                            :ek_amount,
                                            "numerical",
                                            [catalog_id, set_id, keywords, unit, manufacturer_name],
                                            ek_neighborhood_size), :ek_amount)=#
#=
    vk_neighborhood_size = 1
    vk_amount = @trace(embedding_co_occurrence(df,
                                            emb_df,
                                            emb_dict,
                                            [:ek_amount, :lower_bound, :keywords],
                                            ["embedding", "embedding", "embedding"],
                                            :vk_amount,
                                            "numerical",
                                            [ek_amount, lower_bound, keywords],
                                            vk_neighborhood_size), :vk_amount)=#
end

########################################################################################
disable_logging(LogLevel(10))
# Categorical attributes without embeddings
CAT_ATTRS = [#:unit,
            :article_id,
            ]
# Categorical attributes with embeddings
EMB_ATTRS = [:catalog_id,
                #:lower_bound,
                :unit,
                :keywords,
                :manufacturer_name,
                #:ean,
                :set_id,
                ]
# Numerical attributes
NUM_ATTRS = [:ek_amount,
                #:vk_amount,
                ]

# Path of embedding vector files and csv
TSV_PATH_PREFIX = "../data/mercateo/20200227_showcase/ek_100_4069_2000"
CSV_PATH = "../data/mercateo/20200227_showcase/merc_k_0.05_p_0.2.csv"

# read_tsv returns dictionary for tsv pair. Merge dictionaries of all tsv pairs for convenience
CAT_EMBEDDING_DICT = merge([read_tsv("$(TSV_PATH_PREFIX)$(cat_attr)_meta.tsv",
                            "$(TSV_PATH_PREFIX)$(cat_attr)_vec.tsv")
                            for cat_attr in EMB_ATTRS]...)

all_attrs = vcat(CAT_ATTRS, vcat(EMB_ATTRS, NUM_ATTRS))
column_names = reduce(*, [String(attr) * ", " for attr in all_attrs])[1:end-2]
db = SQLite.DB()
ek_table = CSV.File(CSV_PATH) |> SQLite.load!(db, "ek_table")

# load dataset into dataframe
df = SQLite.Query(db, "SELECT $column_names, negative_sample FROM ek_table") |> DataFrame
# load embeddings into dataframe
emb_df = replace_with_emb(df, EMB_ATTRS, vcat(NUM_ATTRS, CAT_ATTRS), CAT_EMBEDDING_DICT)

###################################################################################
# precision recall with negative samples
k = sum(df.negative_sample)

@time emb_sorted_score_df, emb_ns = k_most_improbable(k,
                                            df,
                                            all_attrs,
                                            ek_emb_model,
                                            (df, emb_df, CAT_EMBEDDING_DICT))
emb_precision = convert(Float32, sum(emb_ns)) / convert(Float32, k)

@time emb_NR_sorted_score_df, emb_NR_ns = k_most_improbable(k,
                                            df,
                                            all_attrs,
                                            ek_emb_No_Rules_model,
                                            (df, emb_df, CAT_EMBEDDING_DICT))
emb_NR_precision = convert(Float32, sum(emb_NR_ns)) / convert(Float32, k)

println("---------------")
println("Mercateo model. Precision with embedding:")
println(round(emb_precision; digits=3))

println("Mercateo model. Precision with embedding, no rules:")
println(round(emb_NR_precision; digits=3))
println("---------------")

@time plain_sorted_score_df, plain_ns = k_most_improbable(k,
                                            df,
                                            all_attrs,
                                            ek_plain_model,
                                            (df, ))
plain_precision = convert(Float32, sum(plain_ns)) / convert(Float32, k)
@time plain_NR_sorted_score_df, plain_NR_ns = k_most_improbable(k,
                                            df,
                                            all_attrs,
                                            ek_plain_No_Rules_model,
                                            (df, ))
plain_NR_precision = convert(Float32, sum(plain_NR_ns)) / convert(Float32, k)
println("---------------")
println("Mercateo model. Precision without embedding:")
println(round(plain_precision, digits=3))
println("Mercateo model. Precision without embedding, no rules:")
println(round(plain_NR_precision, digits=3))
println("---------------")


###################################################
# Write coordinates for ROC and PR curves
#=
emb_pr_df, emb_roc_df = get_pr_roc(emb_sorted_score_df, k)
CSV.write("ek_emb_roc.csv", emb_roc_df)
CSV.write("ek_emb_pr.csv", emb_pr_df)
emb_NR_pr_df, emb_NR_roc_df = get_pr_roc(emb_NR_sorted_score_df, k)
CSV.write("ek_emb_NR_roc.csv", emb_NR_roc_df)
CSV.write("ek_emb_NR_pr.csv", emb_NR_pr_df)
plain_pr_df, plain_roc_df = get_pr_roc(plain_sorted_score_df, k)
CSV.write("ek_plain_roc.csv", plain_roc_df)
CSV.write("ek_plain_pr.csv", plain_pr_df)
plain_NR_pr_df, plain_NR_roc_df = get_pr_roc(plain_NR_sorted_score_df, k)
CSV.write("ek_plain_NR_roc.csv", plain_NR_roc_df)
CSV.write("ek_plain_NR_pr.csv", plain_NR_pr_df)
=#
###################################################
# output k most improbable observed tuples
k = 10
@time sorted_score_df, _ = k_most_improbable(k,
                                            df[df.negative_sample.==0, :],
                                            all_attrs,
                                            ek_emb_model,
                                            (df, emb_df, CAT_EMBEDDING_DICT))
for i = 1:k
    println("========================")
    for attr in all_attrs
        attr_val = sorted_score_df[i, attr]
        println("$attr: $attr_val")
    end
    println("log_score: $(sorted_score_df[i, :score])")
end

# Write sorted dataset into csv file
#CSV.write("sorted_merc_data.csv", sorted_score_df)

###################################################

#=# Reduce keyword set into one single keyword
for i in 1:size(df, 1)
    df.keywords[i] = split(df.keywords[i], ", ")[1]
end

mkx = CSV.File("../data/mercateo/mkx_keyword_synonym.csv") |> DataFrame
mkx_dict = Dict(zip(mkx.keyword, mkx.synonym))

for i in 1:size(df, 1)
    kws = split(df.keywords[i], ", ")
    for (j, kw) in enumerate(kws)
        if haskey(mkx_dict, kw)
            df.keywords[i] = mkx_dict[kw]
            break
        elseif j == length(kws)
            println("keyerror $i")
        end
    end
end=#

####################################
