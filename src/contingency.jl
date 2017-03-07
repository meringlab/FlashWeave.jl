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
    make_cum_levels!(cum_levels, Zs, levels)
    z_map_arr = zeros(Int, max_levels_z)
    
    contingency_table!(X, Y, Zs, data, cont_tab, z, cum_levels, z_map_arr, nz)
    
    cont_tab
end


# SPARSE DATA


function update_contingency_table!(entry_vec::Array{Int,1}, cum_levels::Array{Int,1}, z_map_arr::Array{Int,1}, 
    levels_z::Int, cont_tab::Array{Int,3})
    gfp_map = 1
    for j in 3:size(entry_vec, 1)
        gfp_map += entry_vec[j] * cum_levels[j - 2]
    end
    
    level_val = z_map_arr[gfp_map]

    if level_val == 0
        z_map_arr[gfp_map] = levels_z
        level_val = levels_z
        levels_z += 1   
    end
    cont_tab[entry_vec[1] + 1, entry_vec[2] + 1, level_val] += 1
    levels_z
end


function contingency_table!(X::Int, Y::Int, Zs::Array{Int,1}, data::SparseMatrixCSC{Int64,Int64}, row_inds::Vector{Int64},
        vals::Vector{Int64}, cont_tab::Array{Int,3}, cum_levels::Array{Int,1}, z_map_arr::Array{Int,1})
    fill!(cont_tab, 0)
    fill!(z_map_arr, 0)
    data_nnz = nnz(data)
    
    n_vars = 2 + size(Zs, 1)
    nzrange_vec = [nzrange(data, X), nzrange(data, Y), [nzrange(data, Zs[j]) for j in 1:size(Zs, 1)]...]
    
    nzind_vec = Int64[]
    states_vec = Int64[]
    for i in 1:n_vars
        curr_range = nzrange_vec[i]
        (curr_ind, curr_state) = next(curr_range, start(curr_range))
        push!(nzind_vec, curr_ind)
        push!(states_vec, curr_state)
    end

    row_ind_vec = [row_inds[curr_ind] for curr_ind in nzind_vec]
    val_vec = [vals[curr_ind] for curr_ind in nzind_vec]
    entry_vec = zeros(Int, n_vars)
    min_ind = minimum(row_ind_vec)
    num_out_of_bounds = 0
    levels_z = 1
    
    while true
        for j in 1:n_vars
            if row_ind_vec[j] == min_ind
                entry_vec[j] = val_vec[j]
                
                j_state = states_vec[j]
                j_range = nzrange_vec[j]
                if !done(j_range, j_state)
                    (new_j_ind, new_j_state) = next(j_range, j_state)
                    states_vec[j] = new_j_state
                    nzind_vec[j] = new_j_ind
                    row_ind_vec[j] = row_inds[new_j_ind]
                    val_vec[j] = vals[new_j_ind]
                else
                    num_out_of_bounds += 1
                    row_ind_vec[j] = size(data, 1) + 1
                end
            else
                entry_vec[j] = 0
            end
        end
        levels_z = update_contingency_table!(entry_vec, cum_levels, z_map_arr, levels_z, cont_tab)
        min_ind = minimum(row_ind_vec)    
    
        if num_out_of_bounds == n_vars
            break
        end
    end
    
    if z_map_arr[1] != 0
        cont_tab[1, 1, z_map_arr[1]] += size(data, 1) - sum(cont_tab)
    end
    
    levels_z - 1
end


function contingency_table!(X::Int, Y::Int, data::SparseMatrixCSC{Int64,Int64}, row_inds::Vector{Int64}, vals::Vector{Int64}, cont_tab::Array{Int,2})
    fill!(cont_tab, 0)
    
    x_inds = nzrange(data, X)
    y_inds = nzrange(data, Y)
    
    (x_ind, x_state) = next(x_inds, start(x_inds))
    (y_ind, y_state) = next(y_inds, start(y_inds))
    x_row_ind = row_inds[x_ind]
    y_row_ind = row_inds[y_ind]
    x_val = vals[x_ind]
    y_val = vals[y_ind]
    
    min_ind = x_row_ind <= y_row_ind ? x_row_ind : y_row_ind
    num_out_of_bounds = 0
    
    while true
        if x_row_ind == min_ind
            x_entry = x_val + 1
            
            if !done(x_inds, x_state)
                (x_ind, x_state) = next(x_inds, x_state)
                x_row_ind = row_inds[x_ind]
                x_val = vals[x_ind]
            else
                num_out_of_bounds += 1
                x_row_ind = size(data, 1) + 1
            end
        else
            x_entry = 1
        end
        
        if y_row_ind == min_ind
            y_entry = y_val + 1
            
            if !done(y_inds, y_state)
                (y_ind, y_state) = next(y_inds, y_state)
                y_row_ind = row_inds[y_ind]
                y_val = vals[y_ind]
            else
                num_out_of_bounds += 1
                y_row_ind = size(data, 1) + 1
            end
        else
            y_entry = 1
        end
    
        cont_tab[x_entry, y_entry] += 1
        min_ind = min(x_row_ind, y_row_ind)
        
        if num_out_of_bounds >= 2
            break
        end
        
    end
    
    cont_tab[1, 1] += size(data, 1) - length(x_inds) - 1
end


function contingency_table_older!(X::Int, Y::Int, data::SparseMatrixCSC{Int64,Int64}, cont_tab::Array{Int,2})
    fill!(cont_tab, 0)
    x_vec = data[:, X]
    y_vec = data[:, Y]
    x_ind = x_vec.nzind[1]
    y_ind = y_vec.nzind[1]
    x_i = 1
    y_i = 1
    x_val = x_vec.nzval[1]
    y_val = y_vec.nzval[1]
    x_entry = 1
    y_entry = 1
    min_ind = x_ind <= y_ind ? x_ind : y_ind
    num_out_of_bounds = 0
    
    while true
        if x_ind == min_ind
            x_entry = x_val + 1
            x_i += 1
            
            if x_i > nnz(x_vec)
                num_out_of_bounds += 1
                x_ind = x_vec.n + 1
            else
                x_ind = x_vec.nzind[x_i]
                x_val = x_vec.nzval[x_i]
            end
        else
            x_entry = 1
        end
        
        if y_ind == min_ind
            y_entry = y_val + 1
            y_i += 1
            
            if y_i > nnz(y_vec)
                num_out_of_bounds += 1
                y_ind = y_vec.n + 1
            else
                y_ind = y_vec.nzind[y_i]
                y_val = y_vec.nzval[y_i]
            end
        else
            y_entry = 1
        end

        cont_tab[x_entry, y_entry] += 1
        min_ind = min(x_ind, y_ind)
        
        if num_out_of_bounds >= 2
            break
        end
        
    end
    
    cont_tab[1, 1] += x_vec.n - sum(cont_tab)
end


end