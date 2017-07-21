module Preprocessing

export preprocess_data, preprocess_data_default

using StatsBase
#using DataFrames
using FlashWeave.Misc
using FlashWeave.Learning


function mapslices_sparse_nz(f, A::SparseMatrixCSC, dim::Integer=1)
    if dim == 1
        A = A'
    end
    result_vec = zeros(eltype(A), size(A, 2))
    for j in 1:size(A, 2)
        #if dim == 1
        #    col_vec = A[j, :]
        #elseif dim == 2
        #    col_vec = A[:, j]
        #else
        #    error("dim must be 1 or 2")
        #end
        col_vec = A[:, j]
        result_vec[j] = f(col_vec.nzval)
    end
    result_vec
end


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
    Nprod = sum(log.(s[.!z_mask]))
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


function clr{ElType <: AbstractFloat}(X::SparseMatrixCSC{ElType})
    """Specialized version for sparse matrices that always excludes zero entries (thereby no need for pseudo counts)"""
    dest = full(similar(X))
    gmeans_vec = mapslices_sparse_nz(geomean, X', 2)
    broadcast!(/, dest, X, gmeans_vec)
    #dest_sparse = sparse(dest)
    #map!(log, dest_sparse.nzval)
    zmask = dest .== 0.0
    dest = log.(dest)
    dest[zmask] = 0.0
    sparse(dest)
end


function clr{ElType <: AbstractFloat}(X::Matrix{ElType}; pseudo_count::ElType=1e-5, ignore_zeros::Bool=false)
    if pseudo_count == -1 && !ignore_zeros
        X_trans = clr(adaptive_pseudocount(X), pseudo_count=0.0, ignore_zeros=false)
    else
        if pseudo_count != 0.0
            X_trans = X + pseudo_count
        else
            X_trans = X
        end
        center_fun = ignore_zeros ? x -> geomean(x[x .!= pseudo_count]) : geomean

        X_trans = log.(X_trans ./ mapslices(center_fun, X_trans, 2))

        if ignore_zeros
            X_trans[X .== 0.0] = 0.0#minimum(X_trans) - 1
        end
    end

    X_trans
end


function discretize{ElType <: AbstractFloat}(x_vec::Vector{ElType}, n_bins::Integer=3; rank_method::String="tied", disc_method::String="median")
    if disc_method == "median"
        if isempty(x_vec)
            disc_vec = x_vec
        else
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
            disc_vec = map(x -> Int(floor((x) / step)), x_vec)
        end

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


function discretize_nz{ElType <: AbstractFloat}(x_vec::SparseVector{ElType}, n_bins::Integer=3;
        rank_method::String="tied", disc_method::String="median")
    disc_nz_vec = discretize(x_vec.nzval, n_bins-1, rank_method=rank_method, disc_method=disc_method) + 1
    SparseVector(x_vec.n, x_vec.nzind, disc_nz_vec)
end

function discretize_nz{ElType <: AbstractFloat}(x_vec::Vector{ElType}, n_bins::Integer=3; rank_method::String="tied", disc_method::String="median")
    nz_indices = findn(x_vec .!= 0.0)

    if !isempty(nz_indices)
        x_vec_nz = x_vec[nz_indices]
        disc_nz_vec = discretize(x_vec_nz, n_bins-1, rank_method=rank_method, disc_method=disc_method) + 1
        disc_vec = zeros(Int, size(x_vec))
        disc_vec[nz_indices] = disc_nz_vec
    else
        disc_vec = zeros(ElType, size(x_vec))
    end

    disc_vec
end


function discretize{ElType <: AbstractFloat}(X::AbstractMatrix{ElType}; n_bins::Integer=3, nz::Bool=true,
        rank_method::String="tied", disc_method::String="median")
    if nz
        if issparse(X)
            disc_vecs = SparseVector{Int}[]
            for j in 1:size(X, 2)
                push!(disc_vecs, discretize_nz(X[:, j], n_bins, rank_method=rank_method, disc_method=disc_method))
            end
            return hcat(disc_vecs...)
        else
            return mapslices(x -> discretize_nz(x, n_bins, rank_method=rank_method, disc_method=disc_method), X, 1)
        end
    else
        return mapslices(x -> discretize(x, n_bins, rank_method=rank_method, disc_method=disc_method), X, 1)
    end
end


#function factors_to_binary_cols!(data_df::DataFrame, factor_cols::Vector{Symbol})
#    for f_col in factor_cols
#        f_vec = data_df[:, f_col]
#        f_uniques = unique(f_vec)
#        if length(f_uniques) > 2
#            for unique_val in f_uniques
#                new_col = Symbol("$(f_col)_$unique_val")
#                data_df[new_col] = convert(Vector{Int}, f_vec .== unique_val)
#            end
#            delete!(data_df, f_col)
#        end
#    end
#end

iscontinuousnorm(norm::String) = norm == "rows" || startswith(norm, "clr")

function discretize_env!{ElType <: Real}(env_data::Matrix{ElType}, norm, n_bins)
    for i in 1:size(env_data, 2)
        env_vec = env_data[:, i]
        try
            disc_env_vec = convert(Vector{Int}, env_vec)
            if norm == "clr_nz"
                disc_env_vec += 1
            end
            env_vec = convert(Vector{ElType}, disc_env_vec)
        catch InexactError
            if !iscontinuousnorm(norm)
                disc_env_vec = discretize(env_vec, n_bins)
                env_vec = convert(Vector{ElType}, disc_env_vec)
            end
        end
        env_data[:, i] .= env_vec
    end
end

function discretize_env{ElType <: Real}(env_data::SparseMatrixCSC{ElType}, norm, n_bins)
    env_data_dense = full(env_data)
    discretize_env!(env_data_dense, norm, n_bins)
    sparse(env_data_dense)
end


function clrnorm_data(data::AbstractMatrix, norm::String, clr_pseudo_count::AbstractFloat)
    if norm == "clr"
        if issparse(data)
            data = full(data)
        end
        data = convert(Matrix{Float64}, data)
        data = clr(data, pseudo_count=clr_pseudo_count)
    elseif norm == "clr_adapt"
        if issparse(data)
            data = full(data)
        end
        data = convert(Matrix{Float64}, data)
        data = clr(data, pseudo_count=-1.0)
    elseif norm == "clr_nz"
        if issparse(data)
            data = convert(SparseMatrixCSC{Float64}, data)
            data = clr(data)
        else
            data = convert(Matrix{Float64}, data)
            data = clr(data, pseudo_count=0.0, ignore_zeros=true)
        end
    end
    data
end


function rownorm_data(data::AbstractMatrix)
    dest = similar(data, Float64)
    row_sums = sum(data, 2)
    if issparse(data)
        row_sums = sparse(row_sums)
    end
    broadcast!(/, dest, data, row_sums)
    dest
end

function preprocess_data{ElType <: Real}(data::AbstractMatrix{ElType}, norm::String; clr_pseudo_count::AbstractFloat=1e-5, n_bins::Integer=3, rank_method::String="tied",
    disc_method::String="median", verbose::Bool=true, env_cols::Set{Int}=Set{Int}(), make_sparse::Bool=issparse(data), factor_cols::Set{Int}=Set{Int}(),
    prec::Integer=32, filter_data=true, header::Vector{String}=String[])
    if verbose
        println("Removing variables with 0 variance (or equivalently 1 level) and samples with 0 reads")
    end

    if !isempty(env_cols)
        env_data = data[:, sort(collect(env_cols))]
        data = data[:, map(x -> !(x in env_cols), 1:size(data, 2))]
    end

    if filter_data
        unfilt_dims = size(data)
        col_mask = var(data, 1)[:] .> 0.0
        data = data[:, col_mask]
        row_mask = sum(data, 2)[:] .> 0
        data = data[row_mask, :]
        if !isempty(env_cols)
            env_data = env_data[row_mask, :]
        end
        
        if !isempty(header)
            header = header[col_mask]
        end
    end


    if verbose
        println("\tdiscarded ", unfilt_dims[1] - size(data, 1), " samples and ", unfilt_dims[2] - size(data, 2), " variables.")
        println("\nNormalizing data")
    end

    if norm == "rows"
        data = rownorm_data(data)
    elseif startswith(norm, "clr")
        data = clrnorm_data(data, norm, clr_pseudo_count)
    elseif norm == "binary"
        n_bins = 2
        if issparse(data)
            map!(sign, data.nzval)
            data = convert(SparseMatrixCSC{Int}, data)
        else
            data = sign.(data)
            data = convert(Matrix{Int}, data)
        end

        unreduced_vars = size(data, 2)
        bin_mask =  (map(x -> get_levels(data[:, x]), 1:size(data, 2)) .== 2)[:]
        data = data[:, bin_mask]
        
        if !isempty(header)
            header = header[bin_mask]
        end
        if verbose
            println("\tremoved $(unreduced_vars - size(data, 2)) variables with not exactly 2 levels")
        end

    elseif startswith(norm, "binned")
        if startswith(norm, "binned_nz")
            if endswith(norm, "rows")
                data = rownorm_data(data)
            elseif endswith(norm, "clr")
                data = clrnorm_data(data, "clr_nz", 0.0)
            end
            data = discretize(data, n_bins=n_bins, nz=true, rank_method=rank_method, disc_method=disc_method)
        else
            data = discretize(data, n_bins=n_bins, nz=false, rank_method=rank_method, disc_method=disc_method)
        end

        unreduced_vars = size(data, 2)
        bin_mask =  (map(x -> get_levels(data[:, x]), 1:size(data, 2)) .== n_bins)[:]
        data = data[:, bin_mask]
        
        if !isempty(header)
            header = header[bin_mask]
        end

        if verbose
            println("\tremoved $(unreduced_vars - size(data, 2)) variables with less than $n_bins levels")
        end
    else
        error("$norm is no valid normalization method.")
    end

    if !isempty(env_cols)
        if issparse(env_data)
            env_data = discretize_env(env_data, norm, n_bins)
        else
            discretize_env!(env_data, norm, n_bins)
        end
        data = hcat(data, convert(typeof(data), env_data))
    end

    target_base_type_str = iscontinuousnorm(norm) ? "Float" : "Int"
    T = eval(Symbol("$target_base_type_str$prec"))

    if make_sparse
        if !issparse(data)
            data = sparse(data)
        end
        data = convert(SparseMatrixCSC{T}, data)
    else
        if issparse(data)
            data = full(data)
        end
        data = convert(Matrix{T}, data)
    end
    
    if !isempty(header)
        return data, header
    else
        return data
    end
end


function preprocess_data_default(data::AbstractMatrix{ElType}, test_name::String; verbose::Bool=true, make_sparse::Bool=issparse(data), env_cols::Set{Int}=Set{Int}(), factor_cols::Set{Int}=Set{Int}(), prec::Integer=32, header::Vector{String}=String[]) where ElType <: Real
    default_norm_dict = Dict("mi" => "binary", "mi_nz" => "binned_nz_clr", "fz" => "clr_adapt", "fz_nz" => "clr_nz", "mi_expdz" => "binned_nz_clr")
    data = preprocess_data(data, default_norm_dict[test_name]; verbose=verbose, make_sparse=make_sparse, env_cols=env_cols, prec=prec, header=header)
    data
end


end
