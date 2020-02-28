using Gen
using StaticArrays
using LinearAlgebra
using StringDistances

struct EditErrorDistribution <: Gen.Distribution{Symbol} end

const edit_error_distribution = EditErrorDistribution()

function get_word_with_typo(word::String,
                            action::Int64,
                            pos::Int64,
                            chr::Int64)
    # Action: 1 for insertion, 2 for removal, 3 for replacement
    if action==1
        converted_chr = convert(Char, chr+96)
        return word[1:pos-1]*converted_chr*word[pos:end]
    elseif action==2
        return word[1:pos-1]*word[pos+1:end]
    else
        converted_chr = convert(Char, chr+96)
        return replace(word, word[pos]=>converted_chr)
    end
end

function add_typo(n::Int64,
                  intention::Symbol)
  realization = String(intention)
  for _=1:n
    action = Gen.uniform_discrete(1, 3)
    chr = Gen.uniform_discrete(1, 26)
    if action == 1
      pos = Gen.uniform_discrete(1, length(realization) + 1)
    else
      pos = Gen.uniform_discrete(1, length(realization))
    end
    realization = get_word_with_typo(realization, action, pos, chr)
  end
  return Symbol(realization)
end

# TODO Consider implementing TokenMax(Levenshtein())
function Gen.logpdf(::EditErrorDistribution,
                    realization::Symbol,
                    intention::Symbol)
  probs = LinearAlgebra.normalize([0.05^i for i=0:49], 1)
  edit_dist = evaluate(Levenshtein(), String(realization), String(intention))
  d = Gen.Distributions.Categorical(probs)
  i = edit_dist + 1
  Gen.Distributions.logpdf(d, i)
end

function Gen.logpdf_grad(::EditErrorDistribution,
                         realization::Symbol,
                         intention::Symbol)
    (nothing, nothing)
end

function Gen.random(::EditErrorDistribution,
                    intention::Symbol)
  probs = LinearAlgebra.normalize([0.05^i for i=0:49], 1)
  num_typo = rand(Gen.Distributions.Categorical(probs)) - 1

  # add typo
  return add_typo(num_typo, intention)
end

(::EditErrorDistribution)(intention) = Gen.random(EditErrorDistribution(),
                                                 intention)

Gen.has_output_grad(::EditErrorDistribution) = false
Gen.has_argument_grads(::EditErrorDistribution) = (false, false)

export edit_error_distribution
