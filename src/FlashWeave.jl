module FlashWeave

# stdlib
using Distributed, SparseArrays, Statistics, DelimitedFiles, Random
using SharedArrays, LinearAlgebra, Dates

# data structures
using DataStructures
using LightGraphs, SimpleWeightedGraphs

# statistics
using StatsBase, Distributions, Combinatorics

# io
using JSON, HDF5, FileIO

# utilities
import Base: show, names, ==


include("types.jl")
include("io.jl")
include("misc.jl")
include("statfuns.jl")
include("contingency.jl")
include("tests.jl")
include("stackchannels.jl")
include("hiton.jl")
include("interleaved.jl")
include("learning.jl")
include("preprocessing.jl")

export learn_network,
       normalize_data,
       save_network,
       load_network,
       load_data,
       show,
       graph,
       meta_variable_mask


end
