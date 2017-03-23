module Contingency

export contingency_table!, contingency_table

using Cauocc.Misc


function contingency_table!(X::Int, Y::Int, data::Union{SubArray,Matrix{Int64}}, cont_tab::Array{Int,2})
    """2x2"""
    fill!(cont_tab, 0)
    
    for i = 1:size(data, 1)
        x_val = data[i, X] + 1
        y_val = data[i, Y] + 1
        
        cont_tab[x_val, y_val] += 1
    end
end

function contingency_table(X::Int, Y::Int, data::Union{SubArray,Matrix{Int64}}, levels_x::Int, levels_y::Int, nz::Bool=false)
    cont_tab = zeros(Int, levels_x, levels_y)
    contingency_table!(X, Y, data, cont_tab, nz)
        
    cont_tab
end


 
contingency_table(X::Int, Y::Int, data::Union{SubArray,Matrix{Int64}}) = contingency_table(X, Y, data, length(unique(data[:, X])), length(unique(data[:, Y])))
#contingency_table(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Int64}}) = contingency_table(X, Y, Zs, data, length(unique(data[:, X])), length(unique(data[:, Y])))
#contingency_table!(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Int64}}, cont_tab::Array{Int,2}) = contingency_table!(X, Y, data, cont_tab)


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

@generated function contingency_table!{T,N}(X::Int, Y::Int, Zs::T, data::SparseMatrixCSC{Int64,Int64}, row_inds::Vector{Int64},
        vals::Vector{Int64}, cont_tab::Array{Int,3}, cum_levels::Array{Int,1}, z_map_arr::Array{Int,1}, levels::N)
    if T <: Tuple{Int64}
        n_vars = 3
    elseif T <: Tuple{Int64,Int64}
        n_vars = 4
    elseif T <: Tuple{Int64,Int64,Int64}
        n_vars = 5
    else
        return quote error("Sparse matrices are only supported with max_k <= 3") end
    end
        
    if N <: Vector{Int64}
        nz_adjusted = true
    elseif N <: Void
        nz_adjusted = false
    else
        return quote error("Levels needs to be either Vector{Int64} or Void") end
    end
    
    
    expr = quote
        fill!(cont_tab, 0)
        fill!(z_map_arr, -1)
    
        n_rows, n_cols = size(data)
        n_vars = 2 + length(Zs)
        min_row_ind = n_rows
        num_out_of_bounds = 0
        levels_z = 1
        #all_Zs_zero_val = -1
    end
    
    if nz_adjusted
        nz_init_expr = quote
            x_nzadj = levels[X] > 2
            y_nzadj = levels[Y] > 2
            skip_row = false
        end
        append!(expr.args, nz_init_expr.args)
    end
           
    
    var_name_dict = Dict()
    for i in 1:n_vars
        var_name_dict[i] = (Symbol("var_$i"), Symbol("nzi_$i"), Symbol("nzrow_$i"),
                            Symbol("nzval_$i"), Symbol("nzbound_$i"), Symbol("nzentry_$i"))
        var_name, nzi_name, nzrow_name, nzval_name, nzbound_name, nzentry_name = var_name_dict[i]
        
        i_expr = quote
            if $(i) == 1
                $(var_name) = X
            elseif $(i) == 2
                $(var_name) = Y
            else
                $(var_name) = Zs[$(i) - 2]
            end
            
            $(nzi_name) = data.colptr[$(var_name)]
            $(nzrow_name) = row_inds[$(nzi_name)]
            $(nzval_name) = vals[$(nzi_name)]
            
            if $(var_name) != n_cols
                $(nzbound_name) = data.colptr[$(var_name) + 1]
            else
                $(nzbound_name) = nnz(data) + 1
            end
            
            if $(nzi_name) == $(nzbound_name)
                num_out_of_bounds += 1
            elseif $(nzrow_name) < min_row_ind
                min_row_ind = $(nzrow_name)
            end  
        end
        
        
        append!(expr.args, i_expr.args)
    end
    
    loop_expr = quote end
    min_expr = [parse("min_row_ind = min($(join(map(x -> x[3], values(var_name_dict)), ", ")))")]
    
    # expressions for updating variables in the tight inner loop
    for i in 1:n_vars
        var_name, nzi_name, nzrow_name, nzval_name, nzbound_name, nzentry_name = var_name_dict[i]
            
        if nz_adjusted && i == 1
            zero_expr = quote
                if x_nzadj && nzi_1 < nzbound_1
                    skip_row = true
                end
            end
        elseif nz_adjusted && i == 2
            zero_expr = quote
                if y_nzadj && nzi_2 < nzbound_2
                    skip_row = true
                end
            end
        else
            zero_expr = quote end
        end
        
        i_expr = quote
            if $(nzrow_name) == min_row_ind
                $(nzentry_name) = $(nzval_name)
                $(nzi_name) += 1
                
                if $(nzi_name) < $(nzbound_name)
                    $(nzrow_name) = row_inds[$nzi_name]
                    $(nzval_name) = vals[$nzi_name]
                else
                    num_out_of_bounds += 1
                    $(nzrow_name) = n_rows + 1
                end 
            else
                $(zero_expr)
                $(nzentry_name) = 0
            end
        end
        append!(loop_expr.args, i_expr.args)
    end
    
    # compute mapping of the conditioning set
    append!(loop_expr.args, [:(gfp_map = 1)])
    for i_Zs in 1:n_vars-2
        var_name, nzi_name, nzrow_name, nzval_name, nzbound_name, nzentry_name = var_name_dict[i_Zs + 2]
        i_Zs_expr = quote
            gfp_map += $(nzentry_name) * cum_levels[$(i_Zs)]
        end
        append!(loop_expr.args, i_Zs_expr.args)
    end
    
    map_expr = quote
        level_val = z_map_arr[gfp_map]

        if level_val == -1
            z_map_arr[gfp_map] = levels_z
            level_val = levels_z
            levels_z += 1   
        end 
        
        #if gfp_map == 1
        #    all_Zs_zero_val = level_val
        #end
    end
        
    # update contingency table
    X_entry_name = var_name_dict[1][6]
    Y_entry_name = var_name_dict[2][6]
    
    cont_expr = quote
        cont_tab[$(X_entry_name) + 1, $(Y_entry_name) + 1, level_val] += 1
    end
    
    if nz_adjusted
        map_cont_expr = quote
            if !skip_row
                $(map_expr)
                $(cont_expr)
            end
        end
    else
        map_cont_expr = quote
            $(map_expr)
            $(cont_expr)
        end
    end
    append!(loop_expr.args, map_cont_expr.args)
    
    # check breaking criterion
    break_expr = quote
        if num_out_of_bounds >= n_vars
            break
        end
    end
    append!(loop_expr.args, break_expr.args)
    
    # compute new minimum row
    append!(loop_expr.args, min_expr)
    
    # insert loop into main expression
    skip_row_expr = nz_adjusted ? quote skip_row = false end : quote end
    full_loop_expr = quote
        while true
            $(skip_row_expr)
            $(loop_expr)
        end
    end
    append!(expr.args, full_loop_expr.args)
    
    # fill position in contingency table where all variables were 0 and return the conditioning levels
    if !nz_adjusted
        final_expr = quote
            if z_map_arr[1] != -1
                cont_tab[1, 1, z_map_arr[1]] += n_rows - sum(cont_tab)
            end
        end
    else
        final_expr = quote end
    end
    append!(final_expr.args, [:(levels_z - 1)])
    append!(expr.args, final_expr.args)
    
    expr
end



function contingency_table!(X::Int, Y::Int, data::SparseMatrixCSC{Int64,Int64}, row_inds::Vector{Int64}, vals::Vector{Int64},
        cont_tab::Array{Int,2})
    fill!(cont_tab, 0)
    
    n_rows, n_cols = size(data)
    num_out_of_bounds = 0
    
    x_i = data.colptr[X]
    x_row_ind = row_inds[x_i]
    x_val = vals[x_i]
    
    if X != n_cols
        x_bound = data.colptr[X + 1]
    else
        x_bound = nnz(data)
    end
    
    if x_i == x_bound
        num_out_of_bounds += 1
    end
    
    y_i = data.colptr[Y]
    y_row_ind = row_inds[y_i]
    y_val = vals[y_i]
    
    if Y != n_cols
        y_bound = data.colptr[Y + 1]
    else
        y_bound = nnz(data) + 1
    end
    
    if y_i == y_bound
        num_out_of_bounds += 1
    end
    
    min_row_ind = min(x_row_ind, y_row_ind)
    
    while true
        if x_row_ind == min_row_ind
            x_entry = x_val
            x_i += 1
            
            if x_i < x_bound
                x_row_ind = row_inds[x_i]
                x_val = vals[x_i]
            else
                num_out_of_bounds += 1
                x_row_ind = n_rows + 1
            end
        else
            x_entry = 0
        end
        
        if y_row_ind == min_row_ind
            y_entry = y_val
            y_i += 1
            
            if y_i < y_bound
                y_row_ind = row_inds[y_i]
                y_val = vals[y_i]
            else
                num_out_of_bounds += 1
                y_row_ind = n_rows + 1
            end
        else
            y_entry = 0
        end
        
        cont_tab[x_entry + 1, y_entry + 1] += 1
        min_row_ind = min(x_row_ind, y_row_ind)
        
        if num_out_of_bounds >= 2
            break
        end          
    end

    cont_tab[1, 1] += n_rows - sum(cont_tab)
end   
    

end
