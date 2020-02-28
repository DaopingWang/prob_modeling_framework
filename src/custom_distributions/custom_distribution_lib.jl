# Header for custom distributions
#=
include("named_categorical.jl")
include("edit_error_distribution.jl")
include("degenerate_distribution.jl")
include("log_normal.jl")=#
include("half_normal.jl")

using Gen

@dist function uniform_categorical(labels)
    index = uniform_discrete(1, length(labels))
    labels[index]
end

@dist function categorical_named(labels, probs)
    index = categorical(probs)
    labels[index]
end
