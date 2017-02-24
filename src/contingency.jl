module Contingency

export contingency_table!, contingency_table

using Cauocc.Misc


function contingency_table!(X::Int, Y::Int, data::Union{SubArray,Matrix{Int64}}, cont_tab::Array{Int,2}, nz::Bool=false)
    """2x2"""
    fill!(cont_tab, 0)
    
    adj_factor = nz ? 0 : 1
    
    for i = 1:size(data, 1)
        x_val = data[i, X] + adj_factor
        y_val = data[i, Y] + adj_factor
        
        cont_tab[x_val, y_val] += 1
    end
end

function contingency_table(X::Int, Y::Int, data::Union{SubArray,Matrix{Int64}}, levels_x::Int, levels_y::Int, nz::Bool=false)
    cont_tab = zeros(Int, levels_x, levels_y)
    contingency_table!(X, Y, data, cont_tab, nz)
        
    cont_tab
end


 
contingency_table(X::Int, Y::Int, data::Union{SubArray,Matrix{Int64}}, nz::Bool=false) = contingency_table(X, Y, data, length(unique(data[:, X])), length(unique(data[:, Y])), nz)
#contingency_table(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Int64}}) = contingency_table(X, Y, Zs, data, length(unique(data[:, X])), length(unique(data[:, Y])))
#contingency_table!(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Int64}}, cont_tab::Array{Int,2}) = contingency_table!(X, Y, data, cont_tab)


function contingency_table!(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Int64}}, cont_tab::Array{Int, 3},
    z::Vector{Int}, cum_levels::Vector{Int}, z_map_arr::Vector{Int}, nz::Bool=false)
    fill!(cont_tab, 0)
    levels_z = level_map!(Zs, data, z, cum_levels, z_map_arr)
    adj_factor = nz ? 0 : 1

    for i in 1:size(data, 1)
        x_val = data[i, X] + adj_factor
        y_val = data[i, Y] + adj_factor
        z_val = z[i] + 1

        cont_tab[x_val, y_val, z_val] += 1
    end
    
    levels_z
end

# convenience wrapper for three-way contingency tables
function contingency_table(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Int64}}, nz::Bool=false)
    levels = map(x -> length(unique(data[:, x])), 1:size(data, 2))
    max_k = length(Zs)
    levels_x = levels[X]
    levels_y = levels[Y]
    max_levels = maximum(levels)
    max_levels_z = sum([max_levels^(i+1) for i in 1:max_k])
    cont_tab = zeros(Int, levels_x, levels_y, max_levels_z)
    z = zeros(Int, size(data, 1))
    cum_levels = zeros(Int, max_k + 1)
    z_map_arr = zeros(Int, max_levels_z)
    
    contingency_table!(X, Y, Zs, data, cont_tab, z, cum_levels, z_map_arr, nz)
    
    cont_tab
end

"""
function contingency_table!(X::Int, Y::Int, data::Union{SubArray,Matrix{Int64}}, cont_tab::Array{Int,2})
    fill!(cont_tab, 0)
    
    for i = 1:size(data, 1)
        x_val = data[i, X] + 1
        y_val = data[i, Y] + 1
        
        cont_tab[x_val, y_val] += 1
    end
end

function contingency_table(X::Int, Y::Int, data::Union{SubArray,Matrix{Int64}}, levels_x::Int, levels_y::Int)
    cont_tab = zeros(Int, levels_x, levels_y)
    contingency_table(X, Y, data, cont_tab)
        
    cont_tab
end
 
contingency_table(X::Int, Y::Int, data::Union{SubArray,Matrix{Int64}}) = contingency_table(X, Y, data, length(unique(data[:, X])), length(unique(data[:, Y])))
contingency_table(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Int64}}) = contingency_table(X, Y, Zs, data, length(unique(data[:, X])), length(unique(data[:, Y])))
contingency_table!(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Int64}}, cont_tab::Array{Int,2}) = contingency_table!(X, Y, data, cont_tab)


function contingency_table!(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Int64}}, cont_tab::Array{Int, 3},
    z::Vector{Int}, cum_levels::Vector{Int}, z_map_arr::Vector{Int})
    fill!(cont_tab, 0)
    levels_z = level_map!(Zs, data, z, cum_levels, z_map_arr)

    for i in 1:size(data, 1)
        x_val = data[i, X] + 1
        y_val = data[i, Y] + 1
        z_val = z[i] + 1

        cont_tab[x_val, y_val, z_val] += 1
    end
    
    levels_z
end


function contingency_table_nz!(X::Int, Y::Int, data::Union{SubArray,Matrix{Int64}}, cont_tab::Array{Int,2})
    fill!(cont_tab, 0)
    
    for i = 1:size(data, 1)
        x_val = data[i, X]
        y_val = data[i, Y]
        
        cont_tab[x_val, y_val] += 1
    end
end
"""

end