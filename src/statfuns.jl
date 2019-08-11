import Statistics:cor

function fisher_z_transform(p::AbstractFloat, n::Integer, len_z::Integer)
    sample_factor = n - len_z - 3

    if sample_factor > 0
        return (sqrt(sample_factor) / 2.0) * log((1.0 + p) / (1.0 - p))
    else
        return 0.0
    end
end


function oddsratio(ctab::AbstractArray{<:Integer}, nz::Bool=false)
    offset = nz ? 1 : 0
    if ndims(ctab) == 2
        ondiag = (ctab[1 + offset, 1 + offset] * ctab[2 + offset, 2 + offset])
        offdiag = (ctab[1 + offset, 2 + offset] * ctab[2 + offset, 1 + offset])
        @inbounds return (ctab[1 + offset, 1 + offset] * ctab[2 + offset, 2 + offset]) / (ctab[1 + offset, 2 + offset] * ctab[2 + offset, 1 + offset])
    else
        @inbounds oddsratios_per_Zcombo = filter(!isnan, [oddsratio(ctab[:, :, i], false) for i in 1:size(ctab, 3)])
        return isempty(oddsratios_per_Zcombo) ? NaN64 : median(oddsratios_per_Zcombo)
    end
end


function fz_pval(stat::AbstractFloat, n::Int, len_z::Int)
    fz_stat = fisher_z_transform(stat, n, len_z)
    pval = ccdf(Normal(), abs(fz_stat)) * 2.0
    pval
end


function pcor(X::Int, Y::Int, Zs::Tuple{Vararg{Int64,N} where N<:Int}, data::AbstractMatrix{<:Real})
    @inbounds sub_data = @view data[:, [X, Y, Zs...]]

    if size(sub_data, 1) < 1
        return 0.0
    end

    @inbounds p = try
        cov_mat = cov(sub_data)
        inv_mat = pinv(cov_mat)

        var_x = inv_mat[1, 1]
        var_y = inv_mat[2, 2]
        if var_x == 0.0 || var_y == 0.0
            p = 0.0
        else
            p = -inv_mat[1, 2] / sqrt(var_x * var_y)
        end

        # make sure partial correlation coeff stays within bounds
        if p < -1.0
            p = -1.0
        elseif p >= 1.0
            p = 1.0
        end

        return p
    catch
        return 0.0
    end
end


function pcor_rec(X::Int, Y::Int, Zs::Tuple{Vararg{Int64,N} where N<:Int}, cor_mat::AbstractMatrix{ContType},
     pcor_set_dict::Dict{String,Dict{String,ContType}}, cache_result::Bool=true) where ContType<:AbstractFloat

    XY_key = string(X) * "_" * string(Y)
    Zs_key = join(Zs, "_")

    if cache_result && haskey(pcor_set_dict, XY_key) && haskey(pcor_set_dict[XY_key], Zs_key)
        p = pcor_set_dict[XY_key][Zs_key]
    else
        @inbounds if length(Zs) == 1
            Z = Zs[1]

            pXY = cor_mat[X, Y]
            pXZ = cor_mat[X, Z]
            pYZ = cor_mat[Y, Z]
            denom_term = (sqrt(one(ContType) - pXZ^2) * sqrt(one(ContType) - pYZ^2))
            p = denom_term == 0.0 ? 0.0 : (pXY - pXZ * pYZ) / denom_term

        else
            Zs_nZ0 = Zs[1:end-1]
            Z0 = Zs[end]

            pXY_nZ0 = pcor_rec(X, Y, Zs_nZ0, cor_mat, pcor_set_dict, cache_result)
            pXZ0_nZ0 = pcor_rec(X, Z0, Zs_nZ0, cor_mat, pcor_set_dict, cache_result)
            pYZ0_nZ0 = pcor_rec(Y, Z0, Zs_nZ0, cor_mat, pcor_set_dict, cache_result)

            denom_term = sqrt(one(ContType) - pXZ0_nZ0^2) * sqrt(one(ContType) - pYZ0_nZ0^2.0)
            p = denom_term == 0.0 ? 0.0 : (pXY_nZ0 - pXZ0_nZ0 * pYZ0_nZ0) / denom_term
        end


        # make sure partial correlation coeff stays within bounds
        if p < -1.0
            p = -1.0
        elseif p >= 1.0
            p = 1.0
        end


        if cache_result
            if !haskey(pcor_set_dict, XY_key)
                pcor_set_dict[XY_key] = Dict{String, ContType}()
            end

            pcor_set_dict[XY_key][Zs_key] = p
        end
    end

    p
end

function update!(obj::PairMeanObj, x_entry, y_entry)
    obj.sum_x += x_entry
    obj.sum_y += y_entry
    obj.n += 1
end

function update!(obj::PairCorObj, x_entry, y_entry)
    x_entry_norm = x_entry - obj.mean_x
    y_entry_norm = y_entry - obj.mean_y
    obj.cov_xy += x_entry_norm * y_entry_norm
    obj.var_x += x_entry_norm * x_entry_norm
    obj.var_y += y_entry_norm * y_entry_norm
end

function cor(X::Int, Y::Int, data::SparseMatrixCSC{<:Real},
    nz::Bool=false)
    p_mean_obj = PairMeanObj(0.0, 0.0, 0)
    @inbounds iter_apply_sparse_rows!(X, Y, data, update!, p_mean_obj, nz, nz)

    n_obs = nz ? p_mean_obj.n : size(data, 1)

    if n_obs == 0
        return 0.0, 0
    end

    mean_x = p_mean_obj.sum_x / n_obs
    mean_y = p_mean_obj.sum_y / n_obs
    p_cor_obj = PairCorObj(0.0, 0.0, 0.0, mean_x, mean_y)
    @inbounds iter_apply_sparse_rows!(X, Y, data, update!, p_cor_obj, nz, nz)

    if !nz
        z_elems = size(data, 1) - p_mean_obj.n
        p_cor_obj.cov_xy += (-mean_x * -mean_y) * z_elems
        p_cor_obj.var_x += (-mean_x * -mean_x) * z_elems
        p_cor_obj.var_y += (-mean_y * -mean_y) * z_elems
    end

    p = p_cor_obj.cov_xy / sqrt(p_cor_obj.var_x * p_cor_obj.var_y)

    if p > 1.0
        p = 1.0
    elseif p < -1.0
        p = -1.0
    end

    p, n_obs
end


function cor(data::SparseMatrixCSC{<:Real}, nz::Bool)
    n_vars = size(data, 2)
    cor_mat = zeros(Float64, n_vars, n_vars)
    @inbounds Threads.@threads for X in 1:n_vars-1
        for Y in X+1:n_vars
            cor_xy = cor(X, Y, data, nz)
            cor_mat[X, Y] = cor_xy
            cor_mat[Y, X] = cor_xy
        end
    end
    cor_mat
end


function cor_subset!(data::AbstractMatrix{<:Real}, cor_mat::AbstractMatrix{<:AbstractFloat}, vars::AbstractVector{Int})
    n_vars = length(vars)
    """CRITICAL: expects zeros to be trimmed from X and Y in zero-ignoring mode!
    """
    sub_data = @view data[:, vars]
    sub_cors = cor(sub_data)

    @inbounds for i in 1:n_vars-1
        X = vars[i]
        for j in i+1:n_vars
            Y = vars[j]
            cor_xy = sub_cors[i, j]
            cor_val = isnan(cor_xy) ? 0.0 : cor_xy
            cor_mat[X, Y] = cor_val
            cor_mat[Y, X] = cor_val
        end
    end
end


function mi_pval(mi::AbstractFloat, df::Integer, n_obs::Integer)
    g_stat = 2.0 * mi * n_obs
    pval = df > 0 ? ccdf(Chisq(df), g_stat) : 1.0
    pval
end


function mutual_information(ctab::AbstractArray{<:Integer, 3}, levels_x::Integer, levels_y::Integer,
        levels_z::Integer, marg_i::AbstractMatrix{<:Integer}, marg_j::AbstractMatrix{<:Integer}, marg_k::AbstractVector{<:Integer})
    """Note: returns mutual information * number of observations!"""
    fill!(marg_i, 0)
    fill!(marg_j, 0)
    fill!(marg_k, 0)

    # compute marginals
    @inbounds for i in 1:levels_x, j in 1:levels_y, k in 1:levels_z
        marg_i[i, k] += ctab[i, j, k]
        marg_j[j, k] += ctab[i, j, k]
        marg_k[k] += ctab[i, j, k]
    end

    mi_stat = 0.0
    n_obs = sum(ctab)


    # compute mutual information
    @inbounds for i in 1:size(ctab, 1), j in 1:size(ctab, 2), k in 1:size(ctab, 3)
        cell_value = ctab[i, j, k]
        marg_ik = marg_i[i, k]
        marg_jk = marg_j[j, k]

        if cell_value != 0 && marg_ik != 0 && marg_jk != 0
            inner_term = log((marg_k[k] * cell_value) / (marg_ik * marg_jk))
            mi_stat += cell_value * inner_term
        end
    end

    mi_stat / n_obs
end


"""
IMPORTANT NOTE: returns mutual information * number of observations!
(avoids repeated calculation later)
"""
function mutual_information(ctab::AbstractMatrix{<:Integer}, levels_x::Integer, levels_y::Integer,
        marg_i::AbstractVector{<:Integer}, marg_j::AbstractVector{<:Integer})

    fill!(marg_i, 0)
    fill!(marg_j, 0)

    # compute marginals
    @inbounds for i in 1:levels_x, j in 1:levels_y
        marg_i[i] += ctab[i, j]
        marg_j[j] += ctab[i, j]
    end

    mi_stat = 0.0
    n_obs = sum(ctab)

    # compute mutual information
    @inbounds for i in 1:levels_x
        marg_ii = marg_i[i]
        for j in 1:levels_y
            cell_value = ctab[i, j]
            marg_jj = marg_j[j]
            if cell_value != 0 && marg_ii != 0 && marg_jj != 0
                cell_mi = cell_value * log((n_obs * cell_value) / (marg_ii * marg_jj))
                mi_stat += cell_mi
            end
        end
    end

    mi_stat / n_obs
end

## Convenience functions for mutual information

function mutual_information(ctab::AbstractArray{T, 2}) where T<:Integer
    levels_x = size(ctab, 1)
    levels_y = size(ctab, 2)

    ni = zeros(T, levels_x)
    nj = zeros(T, levels_y)

    mutual_information(ctab, levels_x, levels_y, ni, nj)
end


function mutual_information(ctab::AbstractArray{T, 3}) where T<:Integer
    levels_x = size(ctab, 1)
    levels_y = size(ctab, 2)
    levels_z = size(ctab, 3)

    ni = zeros(T, levels_x, levels_z)
    nj = zeros(T, levels_y, levels_z)
    nk = zeros(T, levels_z)

    mutual_information(ctab, levels_x, levels_y, levels_z, ni, nj, nk)
end



function adjust_df(marg_i::AbstractVector{T}, marg_j::AbstractVector{T}, levels_x::Integer, levels_y::Integer) where T<:Integer
    alx = 0
    aly = 0
    @inbounds for i in 1:levels_x
        alx += sign(marg_i[i])
    end
    @inbounds for j in 1:levels_y
        aly += sign(marg_j[j])
    end

    alx = max(1, alx)
    aly = max(1, aly)

    df = (alx - one(T)) * (aly - one(T))

    df
end


function adjust_df(marg_i::AbstractMatrix{T}, marg_j::AbstractMatrix{T}, levels_x::Integer, levels_y::Integer, levels_z::Integer) where T<:Integer
    df = 0
    @inbounds for k in 1:levels_z
        df += adjust_df(marg_i[:, k], marg_j[:, k], levels_x, levels_y)
    end
    df
end

function offset_levels(levels_x::Integer, levels_y::Integer)
    offset_x = levels_x > 2 ? 2 : 1
    offset_y = levels_y > 2 ? 2 : 1
    offset_x, offset_y
end

function nz_adjust_cont_tab(levels_x::Integer, levels_y::Integer, ctab::AbstractArray{<:Integer})
    offset_x, offset_y = offset_levels(levels_x, levels_y)

    if ndims(ctab) == 2
        return @view ctab[offset_x:end, offset_y:end]
    elseif ndims(ctab) == 3
        return @view ctab[offset_x:end, offset_y:end, :]
    else
        error("ctab must have 2 or 3 dimensions (found $(ndims(ctab)) )")
    end
end


"""Accelerated version of that found in MultipleTesting.jl"""
function benjamini_hochberg!(pvals::AbstractVector{T}; alpha::AbstractFloat=0.01,
        m=length(pvals)) where T <: AbstractFloat
    isempty(pvals) && return

    sorted_pval_pairs = filter(x -> x[2] < alpha, collect(enumerate(pvals)))
    isempty(sorted_pval_pairs) && return
    sort!(sorted_pval_pairs, by=x->x[2])

    n_filt = length(sorted_pval_pairs)
    last_index, last_pval = sorted_pval_pairs[end]
    sorted_pval_pairs[end] = (last_index, min(last_pval * m / n_filt, 1.0))

    @inbounds for i in reverse(1:n_filt-1)
        next_adj = sorted_pval_pairs[i+1][2]
        new_adj = sorted_pval_pairs[i][2] * m / i
        min_adj = min(next_adj, new_adj)
        sorted_pval_pairs[i] = (sorted_pval_pairs[i][1], min_adj)
    end

    fill!(pvals, NaN)
    for (i, pval) in sorted_pval_pairs
        pvals[i] = pval
    end
end
