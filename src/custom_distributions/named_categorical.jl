using Gen
using LinearAlgebra

struct NamedCategorical <: Gen.Distribution{Symbol} end

"""
    named_categorical(probs::SVector{N, T},
                      labels::SVector{N, Symbol}) where {N, T<:Real}
Sample an element from `labels` according to the probability masses in
`probs`.  Requires `all(probs .>= 0)` and `sum(probs) == 1.0`.
"""
const named_categorical = NamedCategorical()

function Gen.logpdf(::NamedCategorical,
                    label::Symbol,
                    probs::Array{T, N},
                    labels::Array{Symbol, N}) where {N, T<:Real}
    d = Gen.Distributions.Categorical(probs)
    i = findfirst(==(label), labels)
    Gen.Distributions.logpdf(d, i)
end


function Gen.logpdf_grad(::NamedCategorical,
                         label::Symbol,
                         probs::Array{T, N},
                         labels::Array{Symbol, N}) where {N, T<:Real}
    grad = zeros(length(probs))
    i = findfirst(==(label), labels)
    grad[i] = 1. / probs[i]
    (nothing, nothing, grad)
end

function Gen.random(::NamedCategorical,
                    probs::Array{T, N},
                    labels::Array{Symbol, N}) where {N, T<:Real}
  labels[rand(Gen.Distributions.Categorical(probs))]
end

(::NamedCategorical)(probs, labels) = Gen.random(NamedCategorical(),
                                                 probs, labels)

Gen.has_output_grad(::NamedCategorical) = false
Gen.has_argument_grads(::NamedCategorical) = (false, true)

export named_categorical
