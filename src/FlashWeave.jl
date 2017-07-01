__precompile__()

module FlashWeave

include("misc.jl")
include("statfuns.jl")
include("contingency.jl")
include("tests.jl")
include("stackchannels.jl")
include("learning.jl")
include("preprocessing.jl")

using FlashWeave.Learning

#precompile(LGL, (Matrix{Int64}, ))
#precompile(si_HITON_PC, (Int64, Matrix{Int64}, ))

export LGL, si_HITON_PC

end
