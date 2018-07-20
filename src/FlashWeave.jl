__precompile__()

module FlashWeave

# data structures
using DataStructures
using LightGraphs, SimpleWeightedGraphs

# statistics
using StatsBase, Distributions, Combinatorics

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
       show

function __init__()
   warn_pairs = [(:FileIO, "JLD2"), (:JSON, "BIOM 1.0"), (:HDF5, "BIOM 2.0")]
   for (x, y) in warn_pairs
       isdefined(x) && warn("Package $x was loaded before importing FlashWeave. $y saving/loading will not be available.")
   end
end

end
