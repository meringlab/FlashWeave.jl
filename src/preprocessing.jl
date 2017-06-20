module Preprocessing

export preprocess_data, preprocess_data_default

using StatsBase
using DataFrames
using FlashWeave.Misc
using FlashWeave.Learning


function pseudocount_vars_from_sample_nolog{ElType <: AbstractFloat}(s::Vector{ElType})
    z_mask = s .== 0
    k = sum(z_mask)
    Nprod = prod(s[!z_mask])
    return k, Nprod
end


function adaptive_pseudocount_nolog{ElType <: AbstractFloat}(x1::ElType, s1::Vector{ElType}, s2::Vector{ElType})::ElType
    k, Nprod1 = pseudocount_vars_from_sample(s1)
    n, Nprod2 = pseudocount_vars_from_sample(s2)
    p = length(s1)
    #x2 = nthroot(n-p, (x1^(k-p) * Nprod1) / Nprod2)
    @assert n < p && k < p "samples with all zero abundances are not allowed"
    x2 = ((x1^(k-p) * Nprod1) / Nprod2)^(1/(n-p))
    return x2
end


function pseudocount_vars_from_sample{ElType <: AbstractFloat}(s::Vector{ElType})
    z_mask = s .== 0
    k = sum(z_mask)
    Nprod = sum(log(s[!z_mask]))
    return k, Nprod
end


function adaptive_pseudocount{ElType <: AbstractFloat}(x1::ElType, s1::Vector{ElType}, s2::Vector{ElType})::ElType
    k, Nprod1_log = pseudocount_vars_from_sample(s1)
    n, Nprod2_log = pseudocount_vars_from_sample(s2)
    p = length(s1)
    #x2 = nthroot(n-p, (x1^(k-p) * Nprod1) / Nprod2)
    @assert n < p && k < p "samples with all zero abundances are not allowed"
    #x2_log = (1 / (n-p)) * (log(x1^(k-p)) + Nprod1_log - Nprod2_log)
    x2_log = (1 / (n-p)) * ((k-p)*log(x1) + Nprod1_log - Nprod2_log)
    return exp(x2_log)
end


function adaptive_pseudocount{ElType <: AbstractFloat}(X::Matrix{ElType})
    max_depth_index = findmax(sum(X, 2))[2]
    max_depth_sample::Vector{ElType} = X[max_depth_index, :]
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


function clr{ElType <: AbstractFloat}(X::Matrix{ElType}; pseudo_count::ElType=1e-5, ignore_zeros::Bool=false)
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


function discretize{ElType <: Real}(x_vec::Vector{ElType}, n_bins::Integer=3; rank_method::String="tied", disc_method::String="median")
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


function discretize_nz{ElType <: Real}(x_vec::Vector{ElType}, n_bins::Integer=3, min_elem::ElType=0.0; rank_method::String="tied", disc_method::String="median")
    nz_indices = findn(x_vec .!= min_elem)

    if !isempty(nz_indices)
        x_vec_nz = x_vec[nz_indices]
        disc_nz_vec = discretize(x_vec_nz, n_bins-1, rank_method=rank_method, disc_method=disc_method) + 1
        disc_vec = zeros(Int, size(x_vec))
        disc_vec[nz_indices] = disc_nz_vec
    else
        disc_vec = discretize(x_vec, n_bins-1, rank_method=rank_method, disc_method=disc_method) + 1
    end

    disc_vec
end


function discretize{ElType <: Real}(X::Matrix{ElType}; n_bins::Integer=3, nz::Bool=true, min_elem::ElType=zero(ElType), rank_method::String="tied", disc_method::String="median")
    if nz
        return mapslices(x -> discretize_nz(x, n_bins, min_elem, rank_method=rank_method, disc_method=disc_method), X, 1)
    else
        return mapslices(x -> discretize(x, n_bins, rank_method=rank_method, disc_method=disc_method), X, 1)
    end
end


function factors_to_binary_cols!(data_df::DataFrame, factor_cols::Vector{Symbol})
    for f_col in factor_cols
        f_vec = data_df[:, f_col]
        f_uniques = unique(f_vec)
        if length(f_uniques) > 2
            for unique_val in f_uniques
                new_col = Symbol("$(f_col)_$unique_val")
                data_df[new_col] = convert(Vector{Int}, f_vec .== unique_val)
            end
            delete!(data_df, f_col)
        end
    end
end

iscontinuousnorm(norm::String) = norm == "rows" || startswith(norm, "clr")

function preprocess_data{ElType <: Real}(data::Matrix{ElType}, norm::String; clr_pseudo_count::AbstractFloat=1e-5, n_bins::Integer=3, rank_method::String="tied",
    disc_method::String="median", verbose::Bool=true, env_cols::Set{Int}=Set{Int}(), make_sparse::Bool=false, factor_cols::Set{Int}=Set{Int}(),
    prec::Integer=64)
    if verbose
        println("Removing variables with 0 variance (or equivalently 1 level) and samples with 0 reads")
    end

    if !isempty(env_cols)
        env_data = data[:, sort(collect(env_cols))]
        data = data[:, map(x -> !(x in env_cols), 1:size(data, 2))]
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
        data = convert(Matrix{Int}, data)

        unreduced_vars = size(data, 2)
        data = data[:, (map(x -> get_levels(data[:, x]), 1:size(data, 2)) .== 2)[:]]

        if verbose
            println("\tremoved $(unreduced_vars - size(data, 2)) variables with less than 2 levels")
        end

        if !isempty(env_cols)
            env_data = discretize(env_data, nz=false, n_bins=2)
            env_data = convert(Matrix{Int}, env_data)
        end
    elseif startswith(norm, "binned")
        if startswith(norm, "binned_nz")
            if endswith(norm, "rows")
                data = convert(Matrix{Float64}, data)
                data = data ./ sum(data, 2)
                min_elem = 0.0
            elseif endswith(norm, "clr")
                data = convert(Matrix{Float64}, data)
                data = clr(data, pseudo_count=clr_pseudo_count, ignore_zeros=true)
                min_elem = minimum(data)
            else
                min_elem = 0.0
            end

            data = discretize(data, n_bins=n_bins, nz=true, min_elem=min_elem, rank_method=rank_method, disc_method=disc_method)
        else
            data = discretize(data, n_bins=n_bins, nz=false, rank_method=rank_method, disc_method=disc_method)
        end

        #data = convert(Matrix{Int}, data)

        unreduced_vars = size(data, 2)
        data = data[:, (map(x -> get_levels(data[:, x]), 1:size(data, 2)) .== n_bins)[:]]

        if verbose
            println("\tremoved $(unreduced_vars - size(data, 2)) variables with less than $n_bins levels")
        end

        if !isempty(env_cols)
            env_data = discretize(env_data, nz=false, n_bins=n_bins)
            #env_data = convert(Matrix{Int}, env_data)
        end
    else
        error("$norm is no valid normalization method.")
    end

    if !isempty(env_cols)
        data = hcat(data, convert(typeof(data), env_data))
    end

    target_base_type_str = iscontinuousnorm(norm) ? "Float" : "Int"
    T = eval(Symbol("$target_base_type_str$prec"))
    data = convert(Matrix{T}, data)

    if make_sparse
        data = sparse(data)
    end
    data
end


function preprocess_data_default{ElType <: Real}(data::Matrix{ElType}, test_name::String; verbose::Bool=true, env_cols::Set{Int}=Set{Int}(), make_sparse::Bool=false, factor_cols::Set{Int}=Set{Int}(), prec::Integer=32)
    default_norm_dict = Dict("mi" => "binary", "mi_nz" => "binned_nz_clr", "fz" => "clr_adapt", "fz_nz" => "clr_nz", "mi_expdz" => "binned_nz_clr")
    data = preprocess_data(data, default_norm_dict[test_name]; verbose=verbose, env_cols=env_cols, make_sparse=make_sparse, factor_cols=factor_cols, prec=prec)
    data
end


end
