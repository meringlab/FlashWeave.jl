module Types

using DataStructures
using MetaGraphs
using Combinatorics

export PairMeanObj, PairCorObj,
       AbstractTest, AbstractContTest, AbstractCorTest, ContTest2D, ContTest3D, AbstractNz, Nz, NoNz,
       MiTest, MiTestCond, FzTest, FzTestCond, TestResult,
       reset!, is_zero_adjusted, isdiscrete, iscontinuous,
       HitonState, LGLResult,
       combinations_with_whitelist


#####################
## AUXILLARY TYPES ##
#####################

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

function ZMapper{T<:Integer}(max_k::Integer, max_level::T)
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

function reset!{T<:Integer}(zmap::ZMapper{T})
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

function MiTest{S<:Integer}(levels::Vector{S}, nz::AbstractNz)
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

function MiTestCond{S<:Integer}(levels::Vector{S}, nz::AbstractNz, max_k::Integer)
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

struct HitonState{T}
    phase::String
    state_results::OrderedDict{T,Tuple{Float64,Float64}}
    inter_results::OrderedDict{T,Tuple{Float64,Float64}}
    unchecked_vars::Vector{T}
    state_rejections::Dict{T,Tuple{Tuple,TestResult}}
end

struct LGLResult{T}
    graph::MetaGraph{T,Float64}
    rejections::Dict{T, Dict{T, Tuple{Tuple,TestResult}}}
    unfinished_states::Dict{T, HitonState}
end


#################################
## COMBINATIONS WITH WHITELIST ##
#################################

import Combinatorics:Combinations
import Base:start,next,done

struct CombinationsWL{T,S}
    c::Combinations{T}
    wl::S
end

start(c::CombinationsWL) = start(c.c)
next(c::CombinationsWL, s) = next(c.c, s)

function done(c::CombinationsWL, s)
    if done(c.c, s)
        return true
    else
        (comb, next_s) = next(c.c, s)
        return !(comb[1] in c.wl)
    end
end

function combinations_with_whitelist(a::AbstractVector{T}, wl::AbstractVector{T}, t::Integer) where T <: Integer
    wl_set = Set(wl)
    
    a_wl = copy(wl)
    for e in a
        if !(e in wl_set)
            push!(a_wl, e)
        end
    end
    CombinationsWL(combinations(a_wl, t), wl_set)    
end

end
