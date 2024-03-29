##############
# DENSE DATA #
##############

### 2-way

function contingency_table!(X::Int, Y::Int, data::AbstractMatrix{ElType}, cont_tab::Matrix{<:Integer}) where ElType <: Integer
    """2x2"""
    fill!(cont_tab, 0)

    @inbounds for i = 1:size(data, 1)
        x_val = data[i, X] + one(ElType)
        y_val = data[i, Y] + one(ElType)

        cont_tab[x_val, y_val] += one(ElType)
    end
end


function contingency_table(X::Int, Y::Int, data::AbstractMatrix{<:Integer}, levels_x::Integer, levels_y::Integer)
    cont_tab = zeros(Int, levels_x, levels_y)
    contingency_table!(X, Y, data, cont_tab)

    cont_tab
end


contingency_table(X::Int, Y::Int, data::AbstractMatrix{<:Integer}) = contingency_table(X, Y, data, length(unique(data[:, X])), length(unique(data[:, Y])))

## convenience wrapper for two-way contingency tables
function contingency_table(X::Int, Y::Int, data::SparseArrays.AbstractSparseMatrixCSC{<:Integer}, test_name::String, levels::Vector{<:Integer}=get_levels(data),
    max_vals::Vector{<:Integer}=get_max_vals(data))
    test_obj = make_test_object(test_name, false, max_k=0, levels=levels, max_vals=max_vals, cor_mat=zeros(Float64, 0, 0))
    contingency_table!(X, Y, data, test_obj)
    test_obj.ctab::Matrix{Int}
end

contingency_table(X::Int, Y::Int, data::Matrix{<:Integer}, test_name::String) = contingency_table(X, Y, data)

### 3-way

function contingency_table!(X::Int, Y::Int, Zs::NTuple{N,T} where {N,T<:Integer}, data::AbstractMatrix{ElType}, cont_tab::Array{<:Integer, 3},
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

## convenience wrappers for three-way contingency tables
function contingency_table!(X::Int, Y::Int, Zs::NTuple{N,T} where {N,T<:Integer}, data::Matrix{<:Integer},
    test_obj::ContTest3D)
    z = zeros(eltype(test_obj.levels), size(data, 1))
    contingency_table!(X, Y, Zs, data, test_obj.ctab, z, test_obj.zmap.cum_levels, test_obj.zmap.z_map_arr)
end

function contingency_table(X::Int, Y::Int, Zs::NTuple{N,T} where {N,T<:Integer}, data::AbstractMatrix{<:Integer},
    test_name::String, levels::Vector{<:Integer}=get_levels(data), max_vals::Vector{<:Integer}=get_max_vals(data))
    test_obj = make_test_object(test_name, true, max_k=length(Zs), levels=levels, max_vals=max_vals, cor_mat=zeros(Float64, 0, 0))
    contingency_table!(X, Y, Zs, data, test_obj)
    test_obj.ctab::Array{Int,3}
end


###############
# SPARSE DATA #
###############

### 2-way

# 2-way, optimized for heterogeneous = true
function contingency_table_2d_optim!(X::Int, Y::Int, data::SparseArrays.AbstractSparseMatrixCSC{<:Integer}, test_obj::MiTest{<:Integer, Nz})
    # Initialize a 3x3 zero matrix to hold the contingency table
    fill!(test_obj.ctab, 0)
    rvs = rowvals(data)
    nzvs = nonzeros(data)

    # Get the pointers to the start and end of the non-zero elements in each column
    ptr_X, ptr_Y = data.colptr[X], data.colptr[Y]
    ptr_X_end, ptr_Y_end = data.colptr[X + 1], data.colptr[Y + 1]

    # While there are non-zero elements remaining in either column
    @inbounds while ptr_X < ptr_X_end && ptr_Y < ptr_Y_end
        row_X, row_Y = rvs[ptr_X], rvs[ptr_Y]        
        if row_X == row_Y
            val_X, val_Y = nzvs[ptr_X] + 1, nzvs[ptr_Y] + 1
            test_obj.ctab[val_X, val_Y] += 1
            ptr_X += 1
            ptr_Y += 1
        elseif row_X < row_Y
            ptr_X += 1
        else
            ptr_Y += 1
        end
    end

    return nothing
end

# 2-way, generic fallback method
function contingency_table!(X::Int, Y::Int, data::SparseArrays.AbstractSparseMatrixCSC{<:Integer},
    test_obj::ContTest2D)
    @inbounds if is_zero_adjusted(test_obj)
        X_nz = test_obj.max_vals[X] > 1
        Y_nz = test_obj.max_vals[Y] > 1
    else
        X_nz = Y_nz = false
    end

    if X_nz && Y_nz
        contingency_table_2d_optim!(X, Y, data, test_obj)
    else
        sparse_ctab_backend!((X, Y), data, test_obj, X_nz, Y_nz)
    end
end

### 3-way

"""Make an expression that updates contingency table with zero-nonzero pairs"""
function make_zeroupd_expression(var::Symbol)
    var_other = var == :X ? :Y : :X
    expr = quote
        $(Symbol("val_$(var)")) = nzvs[$(Symbol("ptr_$(var)"))] + 1
        $(Symbol("val_$(var_other)")) = 1
        
        $(make_Zupd_expression(var))

        test_obj.ctab[val_X, val_Y, val_Z] += 1
    end
    return expr
end

"""Make an expression that adds remaining zero-nonzero pairs to contingency table
after main loop has finished"""
function make_zfinish_expression(var::Symbol)
    var_other = var == :X ? :Y : :X
    expr = quote
        $(Symbol("val_$(var_other)")) = 1
        while $(Symbol("ptr_$(var)")) < $(Symbol("ptr_$(var)_end"))
            $(Symbol("val_$(var)")) = nzvs[$(Symbol("ptr_$(var)"))] + 1
            $(Symbol("row_$(var)")) = rvs[$(Symbol("ptr_$(var)"))]
            
            $(make_Zupd_expression(var))
            
            test_obj.ctab[val_X, val_Y, val_Z] += 1
            $(Symbol("ptr_$(var)")) += 1
        end
    end
    return expr
end

"""Make expression for updating Z-related variables"""
function make_Zupd_expression(var::Symbol)
    expr = quote
        while ptr_Z < (ptr_Z_end-1) && row_Z < $(Symbol("row_$(var)"))
            ptr_Z += 1
            row_Z = rvs[ptr_Z]
        end

        if row_Z == $(Symbol("row_$(var)"))
            val_Z = nzvs[ptr_Z] + 1

            if val_Z > levels_z
                levels_z = val_Z
            end
        else
            val_Z = 1
        end
    end
    return expr
end

# 3-way, optimized for max_k = 1 and heterogeneous = true
@generated function contingency_table!(X::Int, Y::Int, Z::Int, data::SparseArrays.AbstractSparseMatrixCSC{<:Integer},
    test_obj::MiTestCond{<:Integer, Nz}, X_nz::Type{TXnz}, Y_nz::Type{TYnz}) where {TXnz <: AbstractNz, TYnz <: AbstractNz}
    expr = quote
        fill!(test_obj.ctab, 0)
        levels_z = 1
        rvs = rowvals(data)
        nzvs = nonzeros(data)

        # Get the pointers to the start and end of the non-zero elements in each column
        ptr_X, ptr_Y, ptr_Z = data.colptr[X], data.colptr[Y], data.colptr[Z]
        ptr_X_end, ptr_Y_end, ptr_Z_end = data.colptr[X + 1], data.colptr[Y + 1], data.colptr[Z + 1]
        val_X = val_Y = val_Z = 1
        row_Z = ptr_Z < ptr_Z_end ? rvs[ptr_Z] : size(data, 1)+1
    end

    # if the other variable is NoNz, insert zero-nonzero or nonzero-zero pairs
    # into ctab
    X_zeroupd_expr = TYnz <: NoNz ? make_zeroupd_expression(:X) : (:())
    Y_zeroupd_expr = TXnz <: NoNz ? make_zeroupd_expression(:Y) : (:())
    
    main_loop_expr = quote
        # While there are non-zero elements remaining in either column
        @inbounds while ptr_X < ptr_X_end && ptr_Y < ptr_Y_end
            row_X, row_Y = rvs[ptr_X], rvs[ptr_Y]
            if row_X == row_Y
                val_X, val_Y = nzvs[ptr_X] + 1, nzvs[ptr_Y] + 1
                $(make_Zupd_expression(:X))
                test_obj.ctab[val_X, val_Y, val_Z] += 1
                ptr_X += 1
                ptr_Y += 1                
            elseif row_X < row_Y
                $(X_zeroupd_expr)
                ptr_X += 1
            else
                $(Y_zeroupd_expr)
                ptr_Y += 1
            end
        end
    end
    append!(expr.args, main_loop_expr.args)
    
    X_zerofinish_expr = TYnz <: NoNz ? make_zfinish_expression(:X) : (:())
    Y_zerofinish_expr = TXnz <: NoNz ? make_zfinish_expression(:Y) : (:())
    finish_expr = quote
        $(X_zerofinish_expr)
        $(Y_zerofinish_expr)
        
        test_obj.zmap.levels_total = levels_z
        
        return nothing
    end

    append!(expr.args, finish_expr.args)

    return expr
end


function contingency_table!(X::Int, Y::Int, Zs::NTuple{N,T} where {N,T<:Integer}, data::SparseArrays.AbstractSparseMatrixCSC{<:Integer},
        test_obj::ContTest3D)
    @inbounds if is_zero_adjusted(test_obj)
        X_nz = test_obj.max_vals[X] > 1
        Y_nz = test_obj.max_vals[Y] > 1
    else
        X_nz = Y_nz = false
    end

    # Special case: max_k = 1 / heterogeneous = true (not applicable if both variables are binary)
    if length(Zs) == 1 && (X_nz || Y_nz)
        X_nz_type = X_nz ? Nz : NoNz
        Y_nz_type = Y_nz ? Nz : NoNz
        contingency_table!(X, Y, Zs[1], data, test_obj, X_nz_type, Y_nz_type)
    # Otherwise use flexible general-purpose backend
    else
        sparse_ctab_backend!((X, Y, Zs...), data, test_obj, X_nz, Y_nz)
    end
end

### generic, flexible 2-way / 3-way sparse backend functions

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

function make_break_expression(T::Type{<:AbstractNz}, i)
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
    return break_expr
end

@generated function sparse_ctab_backend!(cols::NTuple{N,Int}, data::SparseArrays.AbstractSparseMatrixCSC{ElType},
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

        break_expr = make_break_expression(T, i)

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
                $(break_expr)
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

        break_expr = make_break_expression(T, i)

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

