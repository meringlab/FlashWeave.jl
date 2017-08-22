module Contingency

export contingency_table!, contingency_table

using FlashWeave.Misc
using FlashWeave.Types


function contingency_table!{ElType <: Integer}(X::Int, Y::Int, data::AbstractMatrix{ElType}, cont_tab::Matrix{ElType})
    """2x2"""
    fill!(cont_tab, 0)

    for i = 1:size(data, 1)
        x_val = data[i, X] + one(ElType)
        y_val = data[i, Y] + one(ElType)

        cont_tab[x_val, y_val] += one(ElType)
    end
end


function contingency_table{ElType <: Integer}(X::Int, Y::Int, data::AbstractMatrix{ElType}, levels_x::Integer, levels_y::Integer, nz::Bool=false)
    cont_tab = zeros(ElType, levels_x, levels_y)
    contingency_table!(X, Y, data, cont_tab)

    cont_tab
end


contingency_table{ElType <: Integer}(X::Int, Y::Int, data::AbstractMatrix{ElType}) = contingency_table(X, Y, data, length(unique(data[:, X])), length(unique(data[:, Y])))


function contingency_table!{ElType <: Integer}(X::Int, Y::Int, Zs::Vector{Int}, data::AbstractMatrix{ElType}, cont_tab::Array{ElType, 3},
    z::Vector{ElType}, cum_levels::Vector{ElType}, z_map_arr::Vector{ElType})
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

# convenience wrapper for two-way contingency tables
# convenience wrapper for three-way contingency tables
function contingency_table{ElType <: Integer}(X::Int, Y::Int, data::AbstractMatrix{ElType}, test_name::String)
    if issparse(data)
        levels = map(x -> length(unique(data[:, x])), 1:size(data, 2))
        test_obj = make_test_object(test_name, false, max_k=0, levels=levels, cor_mat=zeros(Float64, 0, 0))
        contingency_table!(X, Y, data, test_obj)
        ctab = test_obj.ctab
    else
        ctab = contingency_table(X, Y, data)
    end
    ctab
end


# convenience wrapper for three-way contingency tables
function contingency_table{ElType <: Integer}(X::Int, Y::Int, Zs::Vector{Int}, data::AbstractMatrix{ElType}, test_name::String)
    levels = map(x -> length(unique(data[:, x])), 1:size(data, 2))
    test_obj = make_test_object(test_name, true, max_k=length(Zs), levels=levels, cor_mat=zeros(Float64, 0, 0))
    if issparse(data)
        contingency_table!(X, Y, Zs, data, test_obj)
    else
        z = zeros(eltype(data), size(data, 1))
        contingency_table!(X, Y, Zs, data, test_obj.ctab, z, test_obj.zmap.cum_levels, test_obj.zmap.z_map_arr)
    end
    test_obj.ctab
end


# SPARSE DATA

function contingency_table!{ElType<:Integer}(X::Int, Y::Int, data::SparseMatrixCSC{ElType,Int},
        test_obj::ContTest2D)
    if is_zero_adjusted(test_obj)
        X_nz = test_obj.levels[X] > 2
        Y_nz = test_obj.levels[Y] > 2
    else
        X_nz = Y_nz = false
    end
    
    sparse_ctab_backend!((X, Y), data, test_obj, X_nz, Y_nz)
end

function contingency_table!{ElType<:Integer}(X::Int, Y::Int, Zs::Vector{Int}, data::SparseMatrixCSC{ElType,Int},
        test_obj::ContTest3D)
    if is_zero_adjusted(test_obj)
        X_nz = test_obj.levels[X] > 2
        Y_nz = test_obj.levels[Y] > 2
    else
        X_nz = Y_nz = false
    end
    
    sparse_ctab_backend!((X, Y, Zs...), data, test_obj, X_nz, Y_nz)
end

using FlashWeave.Types

function make_zmap_expression(col_type::Type{NTuple{N,Int}}) where N
    map_expr = quote
        gfp_map = one(ElType)
    end
    for i in 3:N
        val_var = Symbol("val$i")

        i_Zs_expr = quote
            gfp_map += $(val_var) * test_obj.zmap.cum_levels[$(i) - 2]
        end
        append!(map_expr.args, i_Zs_expr.args)
    end
    
    quote
        $(map_expr)
        z_val = test_obj.zmap.z_map_arr[gfp_map]
        if z_val == -1
            z_val = test_obj.zmap.levels_total
            test_obj.zmap.z_map_arr[gfp_map] = z_val
            test_obj.zmap.levels_total += one(ElType)
        end
    end
end

@generated function sparse_ctab_backend!(cols::NTuple{N,Int}, data::SparseMatrixCSC{ElType,Int},
        test_obj::TestType, X_nz::Bool, Y_nz::Bool) where {N, ElType<:Integer, S<:Integer, T<:AbstractNz, TestType<:AbstractContTest{S,T}}
    
    # general init
    # <<----------------
    expr = quote
        n_rows, n_cols = size(data)
        rvals = rowvals(data)
        nzvals = nonzeros(data)
        n_out_of_bounds = 0
        min_ind = n_rows
        break_loop = false
    end
    # ---------------->>
    
    if TestType <: AbstractContTest
        push!(expr.args, :(reset!(test_obj)))        
    end
    
    
    varsymb_dict = Dict()
    
    # initialize variables for each column
    for i in 1:N
        varsymb_dict[i] = (Symbol("col$i"), Symbol("i$i"), Symbol("rowind$i"), Symbol("val$i"), Symbol("bound$i"))
        col_var, i_var, rowind_var, val_var, bound_var = varsymb_dict[i]
        
        # <<----------------
        init_expr = quote
            $(col_var) = cols[$(i)]
            $(i_var) = data.colptr[$(col_var)]
            $(bound_var) = $(col_var) == n_cols ? nnz(data) + 1 : data.colptr[$(col_var) + 1]

            if $(i_var) < $(bound_var)
                $(rowind_var) = rvals[$(i_var)]

                if $(rowind_var) < min_ind
                    min_ind = $(rowind_var)
                end
            else
                $(rowind_var) = n_rows + 1
                n_out_of_bounds += 1
            end
        end
        # -------------->>
        
        append!(expr.args, init_expr.args)
    end
    
    # main loop
    sparse_expr = quote end
    
    for i in 1:N
        col_var, i_var, rowind_var, val_var, bound_var = varsymb_dict[i]
        
        if T <: Nz && i < 3
            adj_name = i == 1 ? Symbol("X_nz") : Symbol("Y_nz")
            skip_expr = quote
                if $(adj_name)
                    skip_row = true
                end
            end
        else
            skip_expr = quote end
        end
        
        if T <: Nz && i < 3
            adj_name = i == 1 ? Symbol("X_nz") : Symbol("Y_nz")
            break_expr = quote
                if $(adj_name)
                    break_loop = true
                end
            end
        else
            break_expr = quote end
        end
        
        # <<----------------        
        i_sparse_expr = quote
            if $(rowind_var) == min_ind
                $(val_var) = nzvals[$(i_var)]
                $(i_var) += 1

                if $(i_var) >= $(bound_var)
                    $(break_expr)
                    n_out_of_bounds += 1
                    $(rowind_var) = n_rows + 1
                else
                    $(rowind_var) = rvals[$(i_var)]
                end
            else
                $(val_var) = zero($(ElType))
                $(skip_expr)
            end

            if $(rowind_var) < next_min_ind
                next_min_ind = $(rowind_var)
            end
        end
        # -------------->>
        if T <: Nz && i > 2
            i_sparse_expr = quote
                if skip_row
                    while $(rowind_var) < next_min_ind
                        $(i_var) += 1

                        if $(i_var) >= $(bound_var)
                            n_out_of_bounds += 1
                            $(rowind_var) = n_rows + 1
                        else
                            $(rowind_var) = rvals[$(i_var)]
                        end
                    end
                else
                    $(i_sparse_expr)
                end
            end
        end
        
        append!(sparse_expr.args, i_sparse_expr.args)
    end
    
    if TestType <: ContTest2D
        val_process_expr = quote
            test_obj.ctab[val1+1, val2+1] += 1
        end
    elseif TestType <: ContTest3D
        zmap_expr = make_zmap_expression(cols)
        
        val_process_expr = quote
            $(zmap_expr)
            test_obj.ctab[val1+1, val2+1, z_val+1] += 1            
        end
    end
    
    # <<----------------
    
    loop_expr = T <: Nz ? quote skip_row = false end : quote end
    if T <: Nz
        val_process_expr = quote
            if !skip_row
                $(val_process_expr)
            end
            
            if break_loop
                break
            end
        end
    end
    
    loop_expr = quote
        while true
            $(loop_expr)
            next_min_ind = n_rows
            $(sparse_expr)
            $(val_process_expr)

            if n_out_of_bounds >= $(N)
                break
            end

            min_ind = next_min_ind
        end
    end
    # ---------------->>
    
    append!(expr.args, loop_expr.args)
    if TestType <: ContTest2D
        push!(expr.args, :(test_obj.ctab[1, 1] += n_rows - sum(test_obj.ctab)))        
    elseif TestType <: ContTest3D
        fill_allzero_expr = quote
            all_zero_obs = n_rows - sum(test_obj.ctab)
            if all_zero_obs > 0
                if test_obj.zmap.z_map_arr[1] != -1 
                    all_zero_z_index = test_obj.zmap.z_map_arr[1] + 1
                else
                    test_obj.zmap.levels_total += one(ElType)
                    all_zero_z_index = test_obj.zmap.levels_total
                end
                test_obj.ctab[1, 1, all_zero_z_index] += all_zero_obs
            end
        end
        append!(expr.args, fill_allzero_expr.args)
    end

    expr
end


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

function update!{ElType <: Integer}(ctab::AbstractMatrix{ElType}, x_entry, y_entry)
    ctab[x_entry+1, y_entry+1] += 1
end


function contingency_table_old!{ElType <: Integer}(X::Int, Y::Int, data::SparseMatrixCSC{ElType},
    cont_tab::Matrix{ElType}, levels_x::Integer, levels_y::Integer, nz::Bool=false)
    fill!(cont_tab, 0)
    iter_apply_sparse_rows!(X, Y, data, update!, cont_tab, nz && levels_x > 2, nz && levels_y > 2)
    if !nz
        cont_tab[1, 1] = size(data, 1) - sum(cont_tab)
    end
end



@generated function contingency_table_old!{T <: Tuple,N,ElType <: Integer}(X::Int, Y::Int, Zs::T, data::SparseMatrixCSC{ElType,Int}, cont_tab::Array{ElType,3},
        cum_levels::Vector{ElType}, z_map_arr::Vector{ElType}, levels::N)
    if T <: Tuple{Int}
        n_vars = 3
    elseif T <: Tuple{Int,Int}
        n_vars = 4
    elseif T <: Tuple{Int,Int,Int}
        n_vars = 5
    else
        return quote error("Sparse matrices are only supported with max_k <= 3") end
    end

    if N <: Vector{ElType}
        nz_adjusted = true
    elseif N <: Void
        nz_adjusted = false
    else
        return quote error("Levels needs to be either Vector{Int} or Void") end
    end


    expr = quote
        fill!(cont_tab, 0)
        fill!(z_map_arr, -1)
        row_inds = rowvals(data)
        vals = nonzeros(data)
        n_rows, n_cols = size(data)
        n_vars = 2 + length(Zs)
        min_row_ind = n_rows
        num_out_of_bounds = 0
        levels_z = one(ElType)
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
                $(nzentry_name) = zero(ElType)
            end
        end
        append!(loop_expr.args, i_expr.args)
    end

    # compute mapping of the conditioning set
    append!(loop_expr.args, [:(gfp_map = one(ElType))])
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
            levels_z += one(ElType)
        end
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
end
