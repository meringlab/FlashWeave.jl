__precompile__()

module FlashWeave

include("types.jl")
include("misc.jl")
include("statfuns.jl")
include("contingency.jl")
include("tests.jl")
include("stackchannels.jl")
include("learning.jl")
include("preprocessing.jl")

using FlashWeave.Learning

export LGL, si_HITON_PC

end
