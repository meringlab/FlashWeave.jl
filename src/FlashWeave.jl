__precompile__()

module FlashWeave

# data structures
using DataStructures
using LightGraphs, SimpleWeightedGraphs

# statistics
using StatsBase, Distributions, Combinatorics
using Clustering

# IO
#using JLD2, FileIO, JSON, HDF5


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
       load_data

end
