module Statfuns

export pcor, fz_pval, mutual_information, mi_pval, adjust_df, oddsratio, nz_adjust_cont_tab

using Distributions


function fisher_z_transform(p::Float64, n::Int, len_z::Int)
    sample_factor = n - len_z - 3

    if sample_factor > 0
        return (sqrt(sample_factor) / 2) * log((1 + p) / (1 - p))
    else
        return 0.0
    end
end


#oddsratio(cont_tab::Union{SubArray,Array{Int,2}}) = (cont_tab[1, 1] * cont_tab[2, 2]) / (cont_tab[1, 2] * cont_tab[2, 1])


function oddsratio(cont_tab::Union{SubArray,Array{Int64}}, nz::Bool=false)
    offset = nz ? 1 : 0
    if ndims(cont_tab) == 2
        return (cont_tab[1 + offset, 1 + offset] * cont_tab[2 + offset, 2 + offset]) / (cont_tab[1 + offset, 2 + offset] * cont_tab[2 + offset, 1 + offset])
    else
        return median([oddsratio(cont_tab[:, :, i], nz) for i in 1:size(cont_tab, 3)])
    end
end


function fz_pval(stat::Float64, n::Int, len_z::Int)
    fz_stat = fisher_z_transform(stat, n, len_z)
    pval = ccdf(Normal(), abs(fz_stat)) * 2
    pval
end



function pcor(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Float64},SparseMatrixCSC{Float64,Int64}})
    sub_data = @view data[:, [X, Y, Zs...]]
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
    p
end


function mi_pval(mi::Float64, df::Int)
    g_stat = 2 * mi
    pval = df > 0 ? ccdf(Chisq(df), g_stat) : 1.0
    pval
end


function mutual_information(cont_tab::Union{SubArray,Array{Int}})
    num_dims = ndims(cont_tab) 
    levels_x = size(cont_tab, 1)
    levels_y = size(cont_tab, 2)
    
    if num_dims == 3
        levels_z = size(cont_tab, 3)
    end
    
    if num_dims == 3
        ni = zeros(Int, levels_x, levels_z)
        nj = zeros(Int, levels_y, levels_z)
        nk = zeros(Int, levels_z)
        
        return mutual_information(cont_tab, levels_x, levels_y, levels_z, ni, nj, nk)
    else
        ni = zeros(Int, levels_x)
        nj = zeros(Int, levels_y)
        
        return mutual_information(cont_tab, levels_x, levels_y, ni, nj)    
    end
  
end


function estimate_expected_dz!(cont_tab::Array, levels_z::Int=0)
    
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


function mutual_information(cont_tab::Union{SubArray,Array{Int,3}}, levels_x::Int, levels_y::Int, levels_z::Int,
        ni::Array{Int,2}, nj::Array{Int,2}, nk::Array{Int, 1}, exp_dz::Bool=false)
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
    #n_obs = sum(cont_tab)
    
    
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

    mi_stat
end


function mutual_information(cont_tab::Union{SubArray,Array{Int,2}}, levels_x::Int, levels_y::Int, ni::Array{Int,1},
        nj::Array{Int,1}, exp_dz::Bool=false)
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
                mi_stat += cell_mi
            end
        end
    end
    
    mi_stat
end


function adjust_df(ni::Array{Int,1}, nj::Array{Int,1}, levels_x::Int, levels_y::Int)
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
    
    df = (alx - 1) * (aly - 1)
    
    df
end


function adjust_df(ni::Array{Int,2}, nj::Array{Int,2}, levels_x::Int, levels_y::Int, levels_z::Int)
    df = 0
    for k in 1:levels_z
        df += adjust_df(ni[:, k], nj[:, k], levels_x, levels_y)
    end
    df
end


function nz_adjust_cont_tab(levels_x::Int64, levels_y::Int64, cont_tab::Array{Int64})
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


"""
function invert_covariance_matrix(cov_mat::Matrix{Float64})
    try
        inv(cov_mat)
    catch exc
        if isa(exc, Base.LinAlg.SingularException)        
            pinv(cov_mat)
        else
            throw(exc)
        end
    end    
end
"""

end