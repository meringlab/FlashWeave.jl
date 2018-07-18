__precompile__()

module FlashWeave

using Iterators

include("types.jl")
include("io.jl")
include("misc.jl")
include("statfuns.jl")
include("contingency.jl")
include("tests.jl")
include("stackchannels.jl")
include("hiton.jl")
include("preclustering.jl")
include("interleaved.jl")
include("learning.jl")
include("preprocessing.jl")

using FlashWeave.Learning
using FlashWeave.Preprocessing

export learn_network, normalize, save_network, load_network, load_data

end
