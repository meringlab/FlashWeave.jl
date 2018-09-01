function contingency_table!(X::Int, Y::Int, data::AbstractMatrix{ElType}, cont_tab::Matrix{<:Integer}) where ElType <: Integer
    """2x2"""
    fill!(cont_tab, 0)

    @inbounds for i = 1:size(data, 1)
        x_val = data[i, X] + one(ElType)
        y_val = data[i, Y] + one(ElType)

        cont_tab[x_val, y_val] += one(ElType)
    end
end


function contingency_table(X::Int, Y::Int, data::AbstractMatrix{<:Integer}, levels_x::Integer, levels_y::Integer, nz::Bool=false)
    cont_tab = zeros(Int, levels_x, levels_y)
    contingency_table!(X, Y, data, cont_tab)

    cont_tab
end


contingency_table(X::Int, Y::Int, data::AbstractMatrix{<:Integer}) = contingency_table(X, Y, data, length(unique(data[:, X])), length(unique(data[:, Y])))


function contingency_table!(X::Int, Y::Int, Zs::Tuple{Vararg{Int64,N} where N<:Int}, data::AbstractMatrix{ElType}, cont_tab::Array{<:Integer, 3},
    z::Vector{<:Integer}, cum_levels::Vector{<:Integer}, z_map_arr::Vector{<:Integer}) where ElType<:Integer
    fill!(cont_tab, 0)
    levels_z = level_map!(Zs, data, z, cum_levels, z_map_arr)

    @inbounds for i in 1:size(data, 1)
        x_val = data[i, X] + one(ElType)
        y_val = data[i, Y] + one(ElType)
        z_val = z[i] + one(ElType)

        cont_tab[x_val, y_val, z_val] += 1
    end

    levels_z
end

## convenience wrapper for two-way contingency tables
function contingency_table(X::Int, Y::Int, data::SparseMatrixCSC{<:Integer}, test_name::String, levels::Vector{<:Integer}=get_levels(data))
    test_obj = make_test_object(test_name, false, max_k=0, levels=levels, cor_mat=zeros(Float64, 0, 0))
    contingency_table!(X, Y, data, test_obj)
    test_obj.ctab::Matrix{Int}
end

contingency_table(X::Int, Y::Int, data::Matrix{<:Integer}, test_name::String) = contingency_table(X, Y, data)


## convenience wrappers for three-way contingency tables
function contingency_table!(X::Int, Y::Int, Zs::Tuple{Vararg{Int64,N} where N<:Int}, data::Matrix{<:Integer},
    test_obj::ContTest3D)
    z = zeros(eltype(test_obj.levels), size(data, 1))
    contingency_table!(X, Y, Zs, data, test_obj.ctab, z, test_obj.zmap.cum_levels, test_obj.zmap.z_map_arr)
end

function contingency_table(X::Int, Y::Int, Zs::Tuple{Vararg{Int64,N} where N<:Int}, data::AbstractMatrix{<:Integer},
    test_name::String, levels::Vector{<:Integer}=get_levels(data))
    test_obj = make_test_object(test_name, true, max_k=length(Zs), levels=levels, cor_mat=zeros(Float64, 0, 0))
    contingency_table!(X, Y, Zs, data, test_obj)
    test_obj.ctab::Array{Int,3}
end



# SPARSE DATA

function contingency_table!(X::Int, Y::Int, data::SparseMatrixCSC{<:Integer},
        test_obj::ContTest2D)

    @inbounds if is_zero_adjusted(test_obj)
        X_nz = test_obj.levels[X] > 2
        Y_nz = test_obj.levels[Y] > 2
    else
        X_nz = Y_nz = false
    end

    sparse_ctab_backend!((X, Y), data, test_obj, X_nz, Y_nz)
end

function contingency_table!(X::Int, Y::Int, Zs::Tuple{Vararg{Int64,N} where N<:Int}, data::SparseMatrixCSC{<:Integer},
        test_obj::ContTest3D)
    @inbounds if is_zero_adjusted(test_obj)
        X_nz = test_obj.levels[X] > 2
        Y_nz = test_obj.levels[Y] > 2
    else
        X_nz = Y_nz = false
    end

    sparse_ctab_backend!((X, Y, Zs...), data, test_obj, X_nz, Y_nz)
end


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

@generated function sparse_ctab_backend!(cols::NTuple{N,Int}, data::SparseMatrixCSC{ElType},
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

            @inbounds if $(i_var) < $(bound_var)
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
            @inbounds if $(rowind_var) == min_ind
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
                            @inbounds $(rowind_var) = rvals[$(i_var)]
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
            @inbounds test_obj.ctab[val1+one(ElType), val2+one(ElType)] += 1
        end
    elseif TestType <: ContTest3D
        zmap_expr = make_zmap_expression(cols)

        val_process_expr = quote
            $(zmap_expr)
            @inbounds test_obj.ctab[val1+one(ElType), val2+one(ElType), z_val+one(ElType)] += 1
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
        push!(expr.args, :(@inbounds test_obj.ctab[1, 1] += n_rows - sum(test_obj.ctab)))
    elseif TestType <: ContTest3D
        fill_allzero_expr = quote
            all_zero_obs = n_rows - sum(test_obj.ctab)
            @inbounds if all_zero_obs > 0
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
