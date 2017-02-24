__precompile__()

module Cauocc

include("statfuns.jl")
include("misc.jl")
include("contingency.jl")
include("tests.jl")
include("stackchannels.jl")
include("learning.jl")

using Cauocc.Learning

export LGL, si_HITON_PC

end