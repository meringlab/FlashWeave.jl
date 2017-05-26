module Preprocessing

export preprocess_data, preprocess_data_default

using StatsBase
using Cauocc.Misc
using Cauocc.Learning

"""
function pw_mi_matrix(data; nz::Bool=false, parallel::String="single")
    workers_local = nprocs() > 1 ? workers_all_local() : true
    test_name = nz ? "mi_nz" : "mi"
    pw_mi_dict = pw_univar_neighbors(data, test_name, parallel=parallel)
    
    mi_mat = zeros(Float64, size(data, 2), size(data, 2))
    
    for var_A in keys(pw_mi_dict)
        for var_B in keys(pw_mi_dict[var_A])
            (stat, pval) = pw_mi_dict[var_A][var_B]
            mi_mat[var_A, var_B] = stat
        end
    end
    mi_mat
end
"""
function pseudocount_vars_from_sample_nolog(s::Vector{Float64})
    z_mask = s .== 0
    k = sum(z_mask)
    Nprod = prod(s[!z_mask])
    return k, Nprod
end


function adaptive_pseudocount_nolog(x1::Float64, s1::Vector{Float64}, s2::Vector{Float64})::Float64
    k, Nprod1 = pseudocount_vars_from_sample(s1)
    n, Nprod2 = pseudocount_vars_from_sample(s2)
    p = length(s1)
    #x2 = nthroot(n-p, (x1^(k-p) * Nprod1) / Nprod2)
    @assert n < p && k < p "samples with all zero abundances are not allowed"
    x2 = ((x1^(k-p) * Nprod1) / Nprod2)^(1/(n-p))
    return x2
end


function pseudocount_vars_from_sample(s::Vector{Float64})
    z_mask = s .== 0
    k = sum(z_mask)
    Nprod = sum(log(s[!z_mask]))
    return k, Nprod
end


function adaptive_pseudocount(x1::Float64, s1::Vector{Float64}, s2::Vector{Float64})::Float64
    k, Nprod1_log = pseudocount_vars_from_sample(s1)
    n, Nprod2_log = pseudocount_vars_from_sample(s2)
    p = length(s1)
    #x2 = nthroot(n-p, (x1^(k-p) * Nprod1) / Nprod2)
    @assert n < p && k < p "samples with all zero abundances are not allowed"
    #x2_log = (1 / (n-p)) * (log(x1^(k-p)) + Nprod1_log - Nprod2_log)
    x2_log = (1 / (n-p)) * ((k-p)*log(x1) + Nprod1_log - Nprod2_log)
    return exp(x2_log)
end


function adaptive_pseudocount(X::Matrix{Float64})
    max_depth_index = findmax(sum(X, 2))[2]
    max_depth_sample::Vector{Float64} = X[max_depth_index, :]
    #pseudo_counts = mapslices(x -> adaptive_pseudocount(1.0, max_depth_sample, x), X, 2)
    min_abund = minimum(X[X .!= 0])
    base_pcount = 1.0#min_abund >= 1 ? 1.0 : min_abund / 10
    pseudo_counts = [adaptive_pseudocount(base_pcount, max_depth_sample, X[x, :]) for x in 1:size(X, 1)]
    
    X_pcount = copy(X)
    
    for i in 1:size(X, 1)
        s_vec = @view X_pcount[i, :]
        s_vec[s_vec .== 0] = pseudo_counts[i]
    end
    X_pcount
end


function clr(X::Matrix; pseudo_count::Float64=1e-5, ignore_zeros=false)
    if pseudo_count == -1 && !ignore_zeros
        X_trans = clr(adaptive_pseudocount(X), pseudo_count=0.0, ignore_zeros=false)
    else
        X_trans = X + pseudo_count
        center_fun = ignore_zeros ? x -> geomean(x[x .!= pseudo_count]) : geomean

        X_trans = log(X_trans ./ mapslices(center_fun, X_trans, 2))

        if ignore_zeros
            X_trans[X .== 0.0] = minimum(X_trans) - 1
        end
    end
    
    X_trans
end


function discretize(x_vec::Vector, n_bins::Int64=3; rank_method::String="tied", disc_method::String="median")
    if disc_method == "median"
        if rank_method == "dense"
            x_vec = denserank(x_vec)
        elseif rank_method == "tied"
            x_vec = tiedrank(x_vec)
        else
            error("$rank_method not a valid ranking method")
        end
    
        x_vec /= maximum(x_vec)
        
        # compute step, add small number to avoid a separate bin for rank 1.0
        step = (1.0 / n_bins) + 1e-5
        #disc_vec = map(x -> Int(floor((x-0.001) / step)), x_vec)
        disc_vec = map(x -> Int(floor((x) / step)), x_vec)
        
    elseif disc_method == "mean"
        if n_bins > 2
            error("disc_method $disc_method only works with 2 bins.")
        end
        
        bin_thresh = mean(x_vec)
        disc_vec = map(x -> x <= bin_thresh ? 0 : 1, x_vec)
    else
        error("$disc_method is not a valid discretization method.")
    end
    
    disc_vec
end
  

function discretize_nz(x_vec::Vector, n_bins::Int64=3, min_elem::Float64=0.0; rank_method::String="tied")
    nz_indices = findn(x_vec .!= min_elem)
    
    if !isempty(nz_indices)
        x_vec_nz = x_vec[nz_indices]
        disc_nz_vec = discretize(x_vec_nz, n_bins-1, rank_method=rank_method) + 1
        disc_vec = zeros(Int64, size(x_vec))
        disc_vec[nz_indices] = disc_nz_vec
    else
        disc_vec = discretize(x_vec, n_bins-1, rank_method=rank_method) + 1
    end
    
    disc_vec        
end


function discretize(X::Matrix; n_bins::Int64=3, nz::Bool=true, min_elem::Float64=0.0, rank_method::String="tied")
    if nz
        return mapslices(x -> discretize_nz(x, n_bins, min_elem, rank_method=rank_method), X, 1)
    else
        return mapslices(x -> discretize(x, n_bins, rank_method=rank_method), X, 1)
    end
end


function preprocess_data(data, norm::String; cluster_sim_threshold::Float64=0.0, clr_pseudo_count::Float64=1e-5, n_bins::Int64=3,
        verbose::Bool=true, skip_cols::Set{Int64}=Set{Int64}())
    if verbose
        println("Removing variables with 0 variance (or equivalently 1 level) and samples with 0 reads")
    end
    
    if !isempty(skip_cols)
        skip_data = data[:, sort(collect(skip_cols))]
        data = data[:, map(x -> !(x in skip_cols), 1:size(data, 2))]
    end
    
    unfilt_dims = size(data)
    col_mask = var(data, 1)[:] .> 0.0
    data = data[:, col_mask]
    row_mask = sum(data, 2)[:] .> 0
    data = data[row_mask, :]
    
    if verbose
        println("\tdiscarded ", unfilt_dims[1] - size(data, 1), " samples and ", unfilt_dims[2] - size(data, 2), " variables.")
        println("\nNormalizing data")
    end
    
    if norm == "rows"
        data = convert(Matrix{Float64}, data)
        data = data ./ sum(data, 2)
    elseif norm == "clr"
        data = convert(Matrix{Float64}, data)
        data = clr(data, pseudo_count=clr_pseudo_count)
    elseif norm == "clr_adapt"
        data = convert(Matrix{Float64}, data)
        data = clr(data, pseudo_count=-1.0)
    elseif norm == "clr_nz"
        data = convert(Matrix{Float64}, data)
        data = clr(data, pseudo_count=clr_pseudo_count, ignore_zeros=true)
        zero_mask = data .== minimum(data)
        data[zero_mask] = 0.0
    elseif norm == "binary"
        data = sign(data)
        data = convert(Matrix{Int64}, data)
        
        unreduced_vars = size(data, 2)
        data = data[:, (map(x -> get_levels(data[:, x]), 1:size(data, 2)) .== 2)[:]]
        
        if verbose
            println("\tremoved $(unreduced_vars - size(data, 2)) variables with less than 2 levels")
        end
    elseif startswith(norm, "binned")
        if startswith(norm, "binned_nz")
            if endswith(norm, "rows")
                data = data ./ sum(data, 2)
                min_elem = 0.0
            elseif endswith(norm, "clr")
                data = clr(data, pseudo_count=clr_pseudo_count, ignore_zeros=true)
                min_elem = minimum(data)
            else
                min_elem = 0.0
            end

            data = discretize(data, n_bins=n_bins, nz=true, min_elem=min_elem)
        else
            data = discretize(data, n_bins=n_bins, nz=false)
        end
            
        data = convert(Matrix{Int64}, data)
        
        unreduced_vars = size(data, 2)
        data = data[:, (map(x -> get_levels(data[:, x]), 1:size(data, 2)) .== n_bins)[:]]
        
        if verbose
            println("\tremoved $(unreduced_vars - size(data, 2)) variables with less than $n_bins levels")
        end
    else
        error("$norm is no valid normalization method.")
    end
    
    if !isempty(skip_cols)
        data = hcat(data, convert(typeof(data), skip_data))
    end
    
    data
end


function preprocess_data_default(data, test_name::String; verbose::Bool=true, skip_cols::Set{Int64}=Set{Int64}())
    default_norm_dict = Dict("mi" => "binary", "mi_nz" => "binned_nz_clr", "fz" => "clr_adapt", "fz_nz" => "clr_nz", "mi_expdz" => "binned_nz_clr")
    data = preprocess_data(data, default_norm_dict[test_name]; verbose=verbose, skip_cols=skip_cols)
    data
end


end