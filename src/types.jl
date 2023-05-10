#####################
## AUXILLARY TYPES ##
#####################

const NbrStatDict = OrderedDict{Int,Tuple{Float64,Float64}}
const StrOrNoth = Union{AbstractString,Nothing}

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
    max_vals::Vector{S}
    marg_i::Vector{S}
    marg_j::Vector{S}
    nz::T
end

function MiTest(levels::Vector{S}, nz::AbstractNz, max_vals::Vector{S}) where S<:Integer
    max_level = maximum(max_vals) + one(S)
    MiTest(zeros(Int, max_level, max_level), levels, max_vals, zeros(S, max_level), zeros(S, max_level), nz)
end

reset!(test_obj::ContTest2D) = fill!(test_obj.ctab, 0)

## 3D CONTINGENCY TESTS ##

abstract type ContTest3D{S,T} <: AbstractContTest{S,T} end
struct MiTestCond{S<:Integer, T<:AbstractNz} <: ContTest3D{S, T}
    ctab::Array{Int, 3}
    zmap::ZMapper{S}
    levels::Vector{S}
    max_vals::Vector{S}
    marg_i::Matrix{S}
    marg_j::Matrix{S}
    marg_k::Vector{S}
    nz::T
end

function MiTestCond(levels::Vector{S}, nz::AbstractNz, max_k::Integer, max_vals::Vector{S}) where S<:Integer
    max_level = maximum(max_vals) + one(S)
    zmap = ZMapper(max_k, max_level)
    ctab = zeros(Int, (max_level, max_level, max_level^max_k))
    marg_i = zeros(S, max_level, max_level^max_k)
    marg_j = copy(marg_i)
    marg_k = zeros(S, max_level^max_k)
    MiTestCond(ctab, zmap, levels, max_vals, marg_i, marg_j, marg_k, nz)
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

const RejDict{T} = Dict{T,Tuple{Tuple{Int64,Vararg{Int64,N} where N},TestResult,Tuple{Int,Float64}}}

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
    n_vars = SimpleWeightedGraphs.nv(inf_results.graph)
    if isnothing(parameters)
        parameters = Dict{Symbol,Any}()
    end

    if isnothing(variable_ids)
        variable_ids = ["X" * string(x) for x in 1:n_vars]
    end

    if isnothing(meta_variable_mask)
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
    n_vars = SimpleWeightedGraphs.nv(G)
    println(io, "$(ne(G)) interactions between $n_vars variables ($(n_vars - n_meta_vars) OTUs and $(n_meta_vars) MVs)\n")

    println(io, "Unfinished variables:")
    n_unf, mean_n_unchecked, mean_frac_unchecked = unchecked_statistics(result)
    if n_unf == 0
        println(io, "none\n")
    else
        println(io, "$n_unf, on average missing $mean_n_unchecked neighbors (mean fraction: $mean_frac_unchecked)\n")
    end

    println(io, "Rejections:")
    println(io, !isempty(rejections(result)) ? "tracked" : "not tracked")
end

###############
## ITERATORS ##
###############

import Base:iterate

struct BNBIterator{M<:AbstractMatrix, T1<:DataType, T2<:NamedTuple, T3<:Tuple}
    X::Int
    Y::Int
    Z_total::Vector{Int}
    data::M
    test_type::T1
    cut_branches::Bool
    test_params::T2 # everything needed for 'make_test_object'
    test_args::T3 # additional args needed to run 'test' (order matters!)
end

struct BNBIteratorState{T1<:AbstractVector, T2<:Tuple,
    T3<:AbstractSet, T4<:Any, T5<:AbstractTest}
    i::Int
    qs::T1
    Zs::T2
    Z_pool::T3
    Z_pool_state::T4
    test_obj::T5
    ref_pval::Float64
end

Base.IteratorSize(::BNBIterator) = Base.SizeUnknown()

function make_test_object(::Type{MiTestCond{S,T}}, max_k_curr, test_params) where {S<:Real, T<:AbstractNz}
    MiTestCond(test_params.levels, T(), max_k_curr, test_params.max_vals)
end

function make_test_object(::Type{FzTestCond{S,T}}, max_k_curr, test_params) where {S<:Real, T<:AbstractNz}
    FzTestCond{S,T}(test_params.cor_mat, Dict{String,Dict{String,S}}(), T(), test_params.cache_pcor)
end

function _test_next(itr, i, qs, Zs, Z, test_obj, ref_pval)
    Zs_test = (Zs..., Z)
    test_res = test(itr.X, itr.Y, Zs_test, itr.data, test_obj, itr.test_args...)

    # length(qs) equals the adjusted max_k, see iterate(itr::BNBIterator)
    # length(qs[i]) >= 2 clause ensures that at least two elements land in the next queue
    # to ensure max_k can be reached when 'cut_branches' is enabled
    if i < length(qs) && test_res.suff_power && (!itr.cut_branches || (test_res.pval > ref_pval || length(qs[i]) < 2))
        qs[i][Z] = test_res.pval
    end
    return test_res, Zs_test
end

function _init_pool(itr::BNBIterator{M,T1,T2,T3}, i, qs) where {M<:AbstractMatrix, T1<:DataType, T2<:NamedTuple, T3<:Tuple}
    if i == 1
        Z_pool = OrderedSet(itr.Z_total)
    else
        Z_pool = OrderedSet(keys(qs[i-1]))
    end

    Z, Z_pool_state = iterate(Z_pool)
    test_obj = make_test_object(itr.test_type, i, itr.test_params)
    return Z_pool, Z_pool_state, Z, test_obj
end

function Base.iterate(itr::BNBIterator)
    max_k = min(itr.test_params.max_k, length(itr.Z_total))
    max_k == 0 && return nothing
    i = 1
    qs = [PriorityQueue{Int,Float64}(Base.Order.Reverse) for i in 1:max_k]
    Zs = ()
    ref_pval = -1.0
    Z_pool, Z_pool_state, Z, test_obj = _init_pool(itr, i, qs)

    next_itr_val = _test_next(itr, i, qs, Zs, Z, test_obj, ref_pval)
    state = BNBIteratorState(i, qs, Zs, Z_pool, Z_pool_state, test_obj, ref_pval)
    return next_itr_val, state
end

function Base.iterate(itr::BNBIterator, state::BNBIteratorState)
    i = state.i
    qs = state.qs
    Zs = state.Zs
    test_obj = state.test_obj
    Z_pool = state.Z_pool
    Z_pool_state = state.Z_pool_state
    ref_pval = state.ref_pval

    Z_pool_ret = iterate(Z_pool, Z_pool_state)

    if !isnothing(Z_pool_ret)
        Z, Z_pool_state = Z_pool_ret
    else
        # find next usable queue (=enough elements to pop one
        # and have at least one more left to populate the next pool)
        while length(qs[i]) < 2
            i -= 1
            i == 0 && return nothing
        end

        if length(Zs) >= i
            Zs = Zs[1:i-1]
        end

        Z_ext, pval = dequeue_pair!(qs[i])
        Zs = (Zs..., Z_ext,)

        ref_pval = itr.cut_branches ? pval : -1.0

        i += 1
        Z_pool, Z_pool_state, Z, test_obj = _init_pool(itr, i, qs)
    end

    next_itr_val = _test_next(itr, i, qs, Zs, Z, test_obj, ref_pval)
    state = BNBIteratorState(i, qs, Zs, Z_pool, Z_pool_state, test_obj, ref_pval)
    return next_itr_val, state
end

######################
## SPECIAL MATRICES ##
######################
import Base:size, show
import SparseArrays: getcolptr, rowvals, nonzeros, nnz, AbstractSparseMatrixCSC

struct SharedSparseMatrixCSC{Tv, Ti} <: AbstractSparseMatrixCSC{Tv, Ti}
    m::Int
    n::Int
    colptr::SharedVector{Ti}
    rowval::SharedVector{Ti}
    nzval::SharedVector{Tv}
end

SharedSparseMatrixCSC(A::SparseMatrixCSC) = SharedSparseMatrixCSC(A.m, A.n, [SharedVector(getproperty(A, x)) for x in (:colptr, :rowval, :nzval)]...)

size(A::SharedSparseMatrixCSC) = (A.m, A.n)
getcolptr(A::SharedSparseMatrixCSC) = A.colptr
#getrowval(A::SharedSparseMatrixCSC) = rowvals(A)
rowvals(A::SharedSparseMatrixCSC) = A.rowval
nonzeros(A::SharedSparseMatrixCSC) = A.nzval
nnz(A::SharedSparseMatrixCSC) = length(A.nzval)

import SparseArrays:_checkbuffers, _goodbuffers

_goodbuffers(S::SharedSparseMatrixCSC) = _goodbuffers(size(S)..., getcolptr(S), rowvals(S), nonzeros(S))
_checkbuffers(S::SharedSparseMatrixCSC) = (@assert _goodbuffers(S); S)
