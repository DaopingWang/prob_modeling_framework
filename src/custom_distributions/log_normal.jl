using Gen
struct LogNormal <: Gen.Distribution{Float64} end

"""
    log_normal(mu::Real, std::Real)
Samples a `Float64` value from a log normal distribution.
"""
const log_normal = LogNormal()

function Gen.logpdf(::LogNormal, x::Real, mu::Real, std::Real)
    diff = log(x) - mu
    var = std * std
    - (diff * diff) / (2.0 * var) - 0.5 * log(2.0 * pi) - log(x * std)
end

function Gen.logpdf_grad(::LogNormal, x::Real, mu::Real, std::Real)
    precision = 1. / (std * std)
    diff = log(x) - mu
    deriv_x = - (1.) / x - precision * (diff / x)
    deriv_mu = diff * precision
    deriv_std = - (1. / std) + (diff * diff) / (std * std * std)
    (deriv_x, deriv_mu, deriv_std)
end

Gen.random(::LogNormal, mu::Real, std::Real) = mu + std * randn()

(::LogNormal)(mu, std) = Gen.random(LogNormal(), mu, std)

Gen.has_output_grad(::LogNormal) = true
Gen.has_argument_grads(::LogNormal) = (true, true)

export log_normal
