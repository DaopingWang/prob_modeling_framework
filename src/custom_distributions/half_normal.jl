using Gen
struct HalfNormal <: Gen.Distribution{Float64} end

"""
    normal(mu::Real, std::Real)
Samples a `Float64` value from a normal distribution.
"""
const half_normal = HalfNormal()

function Gen.logpdf(::HalfNormal, x::Real, mu::Real, std::Real)
    if x < 0
        return -66
    end
    var = std * std
    diff = x - mu
    -(diff * diff)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function Gen.logpdf_grad(::HalfNormal, x::Real, mu::Real, std::Real)
    #TODO
    #=
    precision = 1. / (std * std)
    diff = mu - x
    deriv_x = diff * precision
    deriv_mu = -deriv_x
    deriv_std = -1. / std + (diff * diff) / (std * std * std)
    (deriv_x, deriv_mu, deriv_std)=#
    (nothing, nothing, nothing)
end

Gen.random(::HalfNormal, mu::Real, std::Real) = abs(mu + std * randn())

(::HalfNormal)(mu, std) = Gen.random(HalfNormal(), mu, std)

Gen.has_output_grad(::HalfNormal) = false
Gen.has_argument_grads(::HalfNormal) = (false, false)

export half_normal
