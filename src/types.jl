#####################
## AUXILLARY TYPES ##
#####################

const NbrStatDict = OrderedDict{Int,Tuple{Float64,Float64}}

mutable struct PairMeanObj
    sum_x::Float64
    sum_y::Float64
    n::Int
end

mutable struct PairCorObj
    cov_xy::Float64
    var_x::Float64
    var_y::Float64
    mean_x::Float64
    mean_y::Float64
end

################
## TEST TYPES ##
################

mutable struct ZMapper{T<:Integer}
    z_map_arr::Vector{T}
    cum_levels::Vector{T}
    levels_total::T
end

function ZMapper(max_k::Integer, max_level::T) where T<:Integer
    cum_levels = zeros(T, max_k)
    cum_levels[1] = 1

    for j in 2:max_k
        cum_levels[j] = cum_levels[j - 1] * max_level
    end

    max_mapped_level = max_level
    for j in 1:max_k
        max_mapped_level += max_level * cum_levels[j]
    end
    z_map_arr = fill(-1, max_mapped_level)
    ZMapper{T}(z_map_arr, cum_levels, 0)
end

function reset!(zmap::ZMapper{T}) where T<:Integer
    fill!(zmap.z_map_arr, -1)
    zmap.levels_total = 0
end

abstract type AbstractNz end
struct Nz <: AbstractNz end
struct NoNz <: AbstractNz end

abstract type AbstractTest end
abstract type AbstractContTest{S,T} <: AbstractTest end
abstract type AbstractCorTest{S,T} <: AbstractTest end

is_zero_adjusted(test_obj::AbstractTest) = is_zero_adjusted(typeof(test_obj))
is_zero_adjusted(::Type{TestType}) where {S<:Integer, T<:AbstractNz, TestType<:AbstractContTest{S, T}} = T <: Nz
is_zero_adjusted(::Type{TestType}) where {S<:AbstractFloat, T<:AbstractNz, TestType<:AbstractCorTest{S, T}} = T <: Nz
is_zero_adjusted(test_name::String) = endswith(test_name, "_nz")

isdiscrete(::Type{TestType}) where TestType <: AbstractTest = TestType <: AbstractContTest
isdiscrete(test_obj::AbstractTest) = isdiscrete(typeof(test_obj))
isdiscrete(test_name::String) = test_name in ["mi", "mi_nz"]

iscontinuous(::Type{TestType}) where TestType <: AbstractTest = !isdiscrete(TestType)
iscontinuous(test_obj::AbstractTest) = !isdiscrete(typeof(test_obj))
iscontinuous(test_name::String) = test_name in ["fz", "fz_nz"]


## 2D CONTINGENCY TESTS ##

abstract type ContTest2D{S,T} <: AbstractContTest{S,T} end

struct MiTest{S<:Integer, T<:AbstractNz} <: ContTest2D{S, T}
    ctab::Matrix{Int}
    levels::Vector{S}
    marg_i::Vector{S}
    marg_j::Vector{S}
    nz::T
end

function MiTest(levels::Vector{S}, nz::AbstractNz) where S<:Integer
    max_level = maximum(levels)
    MiTest(zeros(Int, max_level, max_level), levels, zeros(S, max_level), zeros(S, max_level), nz)
end

reset!(test_obj::ContTest2D) = fill!(test_obj.ctab, 0)

## 3D CONTINGENCY TESTS ##

abstract type ContTest3D{S,T} <: AbstractContTest{S,T} end
struct MiTestCond{S<:Integer, T<:AbstractNz} <: ContTest3D{S, T}
    ctab::Array{Int, 3}
    zmap::ZMapper{S}
    levels::Vector{S}
    marg_i::Matrix{S}
    marg_j::Matrix{S}
    marg_k::Vector{S}
    nz::T
end

function MiTestCond(levels::Vector{S}, nz::AbstractNz, max_k::Integer) where S<:Integer
    max_level = maximum(levels)
    zmap = ZMapper(max_k, max_level)
    ctab = zeros(Int, (max_level, max_level, max_level^max_k))
    marg_i = zeros(S, max_level, max_level^max_k)
    marg_j = copy(marg_i)
    marg_k = zeros(S, max_level^max_k)
    MiTestCond(ctab, zmap, levels, marg_i, marg_j, marg_k, nz)
end

function reset!(test_obj::ContTest3D)
    fill!(test_obj.ctab, 0)
    reset!(test_obj.zmap)
end

## FZ TESTS ##

struct FzTest{S<:AbstractFloat, T<:AbstractNz} <: AbstractCorTest{S, T}
    cor_mat::Matrix{S}
    nz::T
end

struct FzTestCond{S<:AbstractFloat, T<:AbstractNz} <: AbstractCorTest{S, T}
    cor_mat::Matrix{S}
    pcor_set_dict::Dict{String,Dict{String,S}}
    nz::T
    cache_pcor::Bool
end

## STATISTICS REPORT ##

struct TestResult
    stat :: Float64
    pval :: Float64
    df :: Int
    suff_power :: Bool
end


##################
## RESULT TYPES ##
##################
const RejDict{T} = Dict{T,Tuple{Tuple,TestResult,Tuple{Int,Float64}}}

struct HitonState{T}
    phase::Char
    state_results::OrderedDict{T,Tuple{Float64,Float64}}
    inter_results::OrderedDict{T,Tuple{Float64,Float64}}
    unchecked_vars::Vector{T}
    state_rejections::RejDict{T}
end

struct LGLResult{T<:Integer}
    graph::SimpleWeightedGraph{T,Float64}
    rejections::Dict{T, RejDict{T}}
    unfinished_states::Dict{T, HitonState{T}}
end

LGLResult(graph::SimpleWeightedGraph{T,Float64}) where T<:Integer = LGLResult(graph, Dict{T, RejDict{T}}(),
                                                                        Dict{T, HitonState{T}}())


struct FWResult{T<:Integer}
    inference_results::LGLResult{T}
    variable_ids::Vector{String}
    meta_variable_mask::BitVector
    parameters::Dict{Symbol,Any}
end

function FWResult(inf_results::LGLResult{T}; variable_ids=nothing, meta_variable_mask=nothing,
    parameters=nothing) where T<:Integer
    n_vars = nv(inf_results.graph)
    if parameters == nothing
        parameters = Dict{Symbol,Any}()
    end

    if variable_ids == nothing
        variable_ids = ["X" * string(x) for x in 1:n_vars]
    end

    if meta_variable_mask == nothing
        meta_variable_mask = falses(n_vars)
    end

    @assert n_vars == length(variable_ids) "variable_ids do not fit number of variables"
    @assert n_vars == length(meta_variable_mask) "meta_variable_mask does not fit number of variables"

    FWResult(inf_results, variable_ids, meta_variable_mask, parameters)
end

FWResult(G::SimpleWeightedGraph; kwargs...) = FWResult(LGLResult(G); kwargs...)


"""
    graph(result::FWResult{T}) -> SimpleWeightedGraph{Int, Float64}

Extract the underlying weighted graph from network results.
"""
graph(result::FWResult{T}) where T<:Integer = result.inference_results.graph
rejections(result::FWResult{T}) where T<:Integer = result.inference_results.rejections
unfinished_states(result::FWResult{T}) where T<:Integer = result.inference_results.unfinished_states

"""
    parameters(result::FWResult{T}) -> Dict{Symbol, Any}

Extract the used parameters from network results.
"""
parameters(result::FWResult{T}) where T<:Integer = result.parameters

"""
    variable_ids(result::FWResult{T}) -> Vector{T}

Extract the IDs/names of all variables (nodes) in the network.
"""
names(result::FWResult{T}) where T<:Integer = result.variable_ids
meta_variable_mask(result::FWResult{T}) where T<:Integer = result.meta_variable_mask
converged(result::FWResult{T}) where T<:Integer = !isempty(result.inference_results.unfinished_states)
==(result1::FWResult{T}, result2::FWResult{S}) where {T<:Integer, S<:Integer} =
    all([f(result1) == f(result2) for f in (graph, names, meta_variable_mask)])

function unchecked_statistics(result::FWResult)
    unf_states_dict = unfinished_states(result)
    if isempty(unf_states_dict)
        0, 0, 0.0
    else
        n_unf = length(unf_states_dict)
        n_unchecked, n_checked = [map(s -> length(getfield(s, fld)), values(unf_states_dict)) for fld in [:unchecked_vars, :state_results]]
        mean_n_unchecked = round(mean(n_unchecked), digits=3)
        mean_frac_unchecked = round(mean(n_unchecked ./ (n_unchecked .+ n_checked)), digits=3)
        n_unf, mean_n_unchecked, mean_frac_unchecked
    end
end


function show(io::IO, result::FWResult{T}) where T<:Integer
    G = graph(result)
    params = parameters(result)
    println(io, "\nMode:")
    if isempty(params)
        println(io, "unknown\n")
    else
        println(io, mode_string([params[key] for key in [:heterogeneous, :sensitive, :max_k]]...), "\n")
    end

    println(io, "Network:")
    n_meta_vars = sum(result.meta_variable_mask)
    n_vars = nv(G)
    println(io, "$(ne(G)) interactions between $n_vars variables ($(n_vars - n_meta_vars) OTUs and $(n_meta_vars) MVs)\n")

    println(io, "Unfinished variables:")
    n_unf, mean_n_unchecked, mean_frac_unchecked = unchecked_statistics(result)
    if n_unf == 0
        println(io, "none\n")
    else
        println(io, "$n_unf, on average missing $mean_n_unchecked neighbors ($mean_frac_unchecked%)\n")
    end

    println(io, "Rejections:")
    println(io, !isempty(rejections(result)) ? "tracked" : "not tracked")
end
