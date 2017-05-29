if nprocs() == 1
    addprocs(1)
end

using Cauocc
using Base.Test

for test_module in ["preprocessing.jl", "misc.jl", "contingency.jl", "statfuns.jl",
                    "tests.jl", "learning.jl"]
    println("\n\nTesting $test_module")
    include(test_module)
end
#include("preprocessing.jl")
#include("misc.jl")
#include("contingency.jl")
#include("statfuns.jl")
#include("tests.jl")
#include("learning.jl")

type HitonState
    phase :: String
    state_results :: Dict{Int,Tuple{Float64,Float64}}
    unchecked_vars :: Vector{Int}
end

function bla(T, data::Matrix{Real}; test_name::String="mi", max_k::Int=3, alpha::Float64=0.01, hps::Int=5,
    pwr::Float64=0.5, FDR::Bool=true, weight_type::String="cond_logpval", whitelist::Set{Int}=Set{Int}(),
        blacklist::Set{Int}=Set{Int}(),
        univar_nbrs::Dict{Int,Tuple{Float64,Float64}}=Dict{Int,Tuple{Float64,Float64}}(), levels::Vector{Int64}=Int64[],
    univar_step::Bool=true, cor_mat::Matrix{Float64}=zeros(Float64, 0, 0),
    pcor_set_dict::Dict{String,Dict{String,Float64}}=Dict{String,Dict{String,Float64}}(),
    prev_state::HitonState=HitonState("S", Dict(), []), debug::Int=0, time_limit::Float64=0.0)
    if debug > 0
        println("Finding neighbors for $T")
    end

    state = HitonState("S", Dict(), [])

    if isdiscrete(test_name)
        if isempty(levels)
            levels = map(x -> get_levels(data[:, x]), 1:size(data, 2))
        end

        if levels[T] < 2
            state.phase = "F"
            state.state_results = Dict{Int,Tuple{Float64,Float64}}()
            state.unchecked_vars = Int64[]
            return state
        end
    else
        levels = Int[]
    end

    if is_zero_adjusted(test_name)
        if !isdiscrete(test_name) || levels[T] > 2
            if issparse(data)
                data = @view data[data[:, T] .!= 0, :]
                #levels = map(x -> get_levels(data[:, x]), 1:size(data, 2))
            else
                data = @view data[data[:, T] .!= 0, :]
                #levels = map(x -> get_levels(data[:, x]), 1:size(data, 2))
            end
        end

    end
end
