__precompile__()

module FlashWeave

# data structures
using DataStructures
using LightGraphs, SimpleWeightedGraphs

# statistics
using StatsBase, Distributions, Combinatorics

# io
using JSON, HDF5, FileIO

# utilities
import Base.show


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

export learn_network,
       normalize_data,
       save_network,
       load_network,
       load_data,
       show,
       graph

# function __init__()
#    warn_items = [(:FileIO, "JLD/JLD2")]
#    for (mod_symbol, format) in warn_items
#        isdefined(mod_symbol) && warn("Package $mod_symbol was loaded before importing FlashWeave. $format will not be available for FlashWeave's IO functions.")
#    end
# end

end
