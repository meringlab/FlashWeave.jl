module Statfuns

export cor_nz, pcor, pcor_rec, cor_subset!, fz_pval, mutual_information, mi_pval, adjust_df, oddsratio, nz_adjust_cont_tab, benjamini_hochberg

import Base:cor

using Distributions

using FlashWeave.Misc


function fisher_z_transform(p::AbstractFloat, n::Integer, len_z::Integer)
    sample_factor = n - len_z - 3
    
    if sample_factor > 0
        return (sqrt(sample_factor) / 2.0) * log((1.0 + p) / (1.0 - p))
    else
        return 0.0
    end
end


function oddsratio{ElType <: Integer}(cont_tab::AbstractArray{ElType}, nz::Bool=false)
    offset = nz ? 1 : 0
    if ndims(cont_tab) == 2
        ondiag = (cont_tab[1 + offset, 1 + offset] * cont_tab[2 + offset, 2 + offset])
        offdiag = (cont_tab[1 + offset, 2 + offset] * cont_tab[2 + offset, 1 + offset])
        return (cont_tab[1 + offset, 1 + offset] * cont_tab[2 + offset, 2 + offset]) / (cont_tab[1 + offset, 2 + offset] * cont_tab[2 + offset, 1 + offset])
    else
        oddsratios_per_Zcombo = filter(!isnan, [oddsratio(cont_tab[:, :, i], false) for i in 1:size(cont_tab, 3)])
        return isempty(oddsratios_per_Zcombo) ? NaN64 : median(oddsratios_per_Zcombo)
    end
end


function fz_pval(stat::AbstractFloat, n::Int, len_z::Int)
    fz_stat = fisher_z_transform(stat, n, len_z)
    pval = ccdf(Normal(), abs(fz_stat)) * 2.0
    pval
end


function pcor{ElType <: AbstractFloat}(X::Int, Y::Int, Zs::AbstractVector{Int}, data::AbstractMatrix{ElType})
    sub_data = @view data[:, [X, Y, Zs...]]

    if size(sub_data, 1) < 1
        return 0.0
    end

    p = try
        cov_mat = cov(sub_data)
        #println("$X, $Y, $Zs, $cov_mat")
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


function pcor_rec{ElType <: AbstractFloat}(X::Int, Y::Int, Zs::AbstractVector{Int}, cor_mat::AbstractMatrix{ElType}, pcor_set_dict::Dict{String,Dict{String,ElType}}, cache_result::Bool=true)
    XY_key = join((X, Y), "_")
    Zs_key = join(Zs, "_")

    if haskey(pcor_set_dict, XY_key) && haskey(pcor_set_dict[XY_key], Zs_key)
        p = pcor_set_dict[XY_key][Zs_key]
    else
        if length(Zs) == 1
            Z = Zs[1]

            pXY = cor_mat[X, Y]
            pXZ = cor_mat[X, Z]
            pYZ = cor_mat[Y, Z]
            denom_term = (sqrt(one(ElType) - pXZ^2) * sqrt(one(ElType) - pYZ^2))
            p = denom_term == 0.0 ? 0.0 : (pXY - pXZ * pYZ) / denom_term

        else
            Zs_nZ0 = Zs[1:end-1]
            Z0 = Zs[end]

            pXY_nZ0 = pcor_rec(X, Y, Zs_nZ0, cor_mat, pcor_set_dict)
            pXZ0_nZ0 = pcor_rec(X, Z0, Zs_nZ0, cor_mat, pcor_set_dict)
            pYZ0_nZ0 = pcor_rec(Y, Z0, Zs_nZ0, cor_mat, pcor_set_dict)

            denom_term = sqrt(one(ElType) - pXZ0_nZ0^2) * sqrt(one(ElType) - pYZ0_nZ0^2.0)
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
                pcor_set_dict[XY_key] = Dict{String, ElType}()
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

function cor{ElType <: AbstractFloat}(X::Int, Y::Int, data::SparseMatrixCSC{ElType},
    nz::Bool=false)
    p_mean_obj = PairMeanObj(0.0, 0.0, 0)
    iter_apply_sparse_rows!(X, Y, data, update!, p_mean_obj, nz, nz)

    n_obs = nz ? p_mean_obj.n : size(data, 1)

    if n_obs == 0
        return 0.0, 0
    end

    mean_x = p_mean_obj.sum_x / n_obs
    mean_y = p_mean_obj.sum_y / n_obs
    p_cor_obj = PairCorObj(0.0, 0.0, 0.0, mean_x, mean_y)
    iter_apply_sparse_rows!(X, Y, data, update!, p_cor_obj, nz, nz)

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


function cor{ElType <: AbstractFloat}(data::SparseMatrixCSC{ElType}, nz::Bool)
    n_vars = size(data, 2)
    cor_mat = zeros(Float64, n_vars, n_vars)
    Threads.@threads for X in 1:n_vars-1
        for Y in X+1:n_vars
            cor_xy = cor(X, Y, data, nz)
            cor_mat[X, Y] = cor_xy
            cor_mat[Y, X] = cor_xy
        end
    end
    cor_mat
end


function cor_subset!{ElType <: AbstractFloat}(data::AbstractMatrix{ElType}, cor_mat::AbstractMatrix{ElType}, vars::AbstractVector{Int})
    n_vars = length(vars)
    """CRITICAL: expects zeros to be trimmed from X and Y in zero-ignoring mode!
    """
    #if nz
    #    sub_data = @view data[data[:, vars[2]] .!= 0.0, vars]
    #else
    #    sub_data = @view data[:, vars]
    #end
    #sub_cors = cor(sub_data)
    sub_data = @view data[:, vars]
    sub_cors = cor(sub_data)

    for i in 1:n_vars-1
        X = vars[i]
        for j in i+1:n_vars
            Y = vars[j]
            #cor_xy = cor(X, Y, data, nz)
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


function mutual_information{ElType <: Integer}(cont_tab::AbstractArray{ElType})
    num_dims = ndims(cont_tab)
    levels_x = size(cont_tab, 1)
    levels_y = size(cont_tab, 2)

    if num_dims == 3
        levels_z = size(cont_tab, 3)
    end

    if num_dims == 3
        ni = zeros(ElType, levels_x, levels_z)
        nj = zeros(ElType, levels_y, levels_z)
        nk = zeros(ElType, levels_z)

        return mutual_information(cont_tab, levels_x, levels_y, levels_z, ni, nj, nk)
    else
        ni = zeros(ElType, levels_x)
        nj = zeros(ElType, levels_y)

        return mutual_information(cont_tab, levels_x, levels_y, ni, nj)
    end

end


function estimate_expected_dz!{ElType <: AbstractFloat}(cont_tab::AbstractArray{ElType, 3}, levels_z::Integer=0)

    if ndims(cont_tab) == 3
        for i in 1:levels_z
            estimate_expected_dz!(cont_tab[:, :, i])
        end
    else
        num_dz = cont_tab[1, 1]

        while true
            n_obs = sum(cont_tab)
            #cont_tab_rel = cont_tab ./ n_obs
            prev_exp = cont_tab[1, 1]
            new_exp = Int(round((sum(cont_tab[1, :]) / n_obs) * (sum(cont_tab[:, 1]) / n_obs) * n_obs))
            cont_tab[1, 1] = new_exp

            if new_exp == prev_exp
                break
            else
                prev_exp = new_exp
            end
        end
    end
end


function mutual_information{ElType <: Integer}(cont_tab::AbstractArray{ElType, 3}, levels_x::Integer, levels_y::Integer, levels_z::Integer,
        ni::AbstractMatrix{ElType}, nj::AbstractMatrix{ElType}, nk::AbstractVector{ElType}, exp_dz::Bool=false)
    """Note: returns mutual information * number of observations!"""
    #if reset_marginals
    fill!(ni, 0)
    fill!(nj, 0)
    fill!(nk, 0)

    if exp_dz
        estimate_expected_dz!(cont_tab, levels_z)
    end
    #println("cont tab:", cont_tab)

    # compute marginals
    for i in 1:levels_x, j in 1:levels_y, k in 1:levels_z
        #println("i, j, k: $i $j $k")
        ni[i, k] += cont_tab[i, j, k]
        nj[j, k] += cont_tab[i, j, k]
        nk[k] += cont_tab[i, j, k]
    end

    mi_stat = 0.0
    n_obs = sum(cont_tab)


    # compute mutual information
    for i in 1:size(cont_tab, 1), j in 1:size(cont_tab, 2), k in 1:size(cont_tab, 3)

        #if exp_dz && i == 1 && j == 1
        #    continue
        #end

        cell_value = cont_tab[i, j, k]
        nik = ni[i, k]
        njk = nj[j, k]

        if cell_value != 0 && nik != 0 && njk != 0
            inner_term = log((nk[k] * cell_value) / (nik * njk))
            mi_stat += cell_value * inner_term
        end
    end

    mi_stat / n_obs
end


function mutual_information{ElType <: Integer}(cont_tab::AbstractMatrix{ElType}, levels_x::Integer, levels_y::Integer, ni::AbstractVector{ElType},
        nj::AbstractVector{ElType}, exp_dz::Bool=false)
    """Note: returns mutual information * number of observations!"""
    fill!(ni, 0)
    fill!(nj, 0)

    if exp_dz
        estimate_expected_dz!(cont_tab)
    end

    # compute marginals
    for i in 1:levels_x, j in 1:levels_y
        ni[i] += cont_tab[i, j]
        nj[j] += cont_tab[i, j]
    end

    mi_stat = 0.0
    n_obs = sum(cont_tab)

    # compute mutual information
    for i in 1:levels_x
        nii = ni[i]
        for j in 1:levels_y
            #if exp_dz && i == 1 && j == 1
            #    continue
            #end

            cell_value = cont_tab[i, j]
            njj = nj[j]
            if cell_value != 0 && nii != 0 && njj != 0
                cell_mi = cell_value * log((n_obs * cell_value) / (nii * njj))
                #cell_mi = cell_value * ((log(n_obs) + log(cell_value)) - (log(nii) + log(njj)))
                mi_stat += cell_mi
            end
        end
    end

    mi_stat / n_obs
end


function adjust_df{ElType <: Integer}(ni::AbstractVector{ElType}, nj::AbstractVector{ElType}, levels_x::Integer, levels_y::Integer)
    alx = 0
    aly = 0
    for i in 1:levels_x
        alx += sign(ni[i])
    end
    for j in 1:levels_y
        aly += sign(nj[j])
    end

    alx = max(1, alx)
    aly = max(1, aly)

    df = (alx - one(ElType)) * (aly - one(ElType))

    df
end


function adjust_df{ElType <: Integer}(ni::AbstractMatrix{ElType}, nj::AbstractMatrix{ElType}, levels_x::Integer, levels_y::Integer, levels_z::Integer)
    df = 0
    for k in 1:levels_z
        df += adjust_df(ni[:, k], nj[:, k], levels_x, levels_y)
    end
    df
end


function nz_adjust_cont_tab{ElType <: Integer}(levels_x::Integer, levels_y::Integer, cont_tab::AbstractArray{ElType})
    offset_x = levels_x > 2 ? 2 : 1
    offset_y = levels_y > 2 ? 2 : 1

    if ndims(cont_tab) == 2
        return @view cont_tab[offset_x:end, offset_y:end]
    elseif ndims(cont_tab) == 3
        return @view cont_tab[offset_x:end, offset_y:end, :]
    else
        error("cont_tab must have 2 or 3 dimensions (found $(ndims(cont_tab)) )")
    end
end


function benjamini_hochberg{ElType <: AbstractFloat}(pvals::AbstractVector{ElType})
    """Accelerated version of that found in MultipleTesting.jl"""
    m = length(pvals)

    sorted_pval_tuples::Vector{Tuple{Int,ElType}} = collect(zip(1:length(pvals), pvals))
    sort!(sorted_pval_tuples, by=x->x[2])

    for i in reverse(1:m-1)
        next_adj = sorted_pval_tuples[i+1][2]
        new_adj = sorted_pval_tuples[i][2] * m / i
        min_adj = min(next_adj, new_adj)
        sorted_pval_tuples[i] = (sorted_pval_tuples[i][1], min_adj)
    end

    sort!(sorted_pval_tuples, by=x->x[1])
    return [x[2] for x in sorted_pval_tuples]
end


end
