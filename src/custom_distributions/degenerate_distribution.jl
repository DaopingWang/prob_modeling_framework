using Gen
using LinearAlgebra

struct DegenerateDistribution <: Distribution{Symbol} end

"""
    degenerate_distribution(probs::AbstractArray{U, 1}) where {U <: Real}
Given a vector of probabilities `probs` where `sum(probs) = 1`, sample an `Int` `i` from the set {1, 2, .., `length(probs)`} with probability `probs[i]`.
"""
const degenerate_distribution = DegenerateDistribution()

function Gen.logpdf(::DegenerateDistribution, retval::Symbol, x::Symbol)
    if retval === x
        return log(1.)
    else
        return 0.
    end
end

function Gen.logpdf_grad(::DegenerateDistribution, retval::Symbol, x::Symbol)
    if retval === x
        return (nothing, 1.)
    else
        return (nothing, 0.)
    end
end

function Gen.random(::DegenerateDistribution, x::Symbol)
    return x
end

(::DegenerateDistribution)(x) = random(DegenerateDistribution(), x)

Gen.has_output_grad(::DegenerateDistribution) = false
Gen.has_argument_grads(::DegenerateDistribution) = true

export degenerate_distribution
