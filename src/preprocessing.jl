module Preprocessing

export preprocess_data

using StatsBase
using Cauocc.Misc
using Cauocc.Learning

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


function cluster_data(data, sim_type::String="pearson", cluster_sim_threshold::Float64=0.8, parallel="single",
    ordering="size")
    if sim_type == "pearson"
        sim_mat = abs(cor(data))
    elseif sim_type == "mi"
        sim_mat = pw_mi_matrix(data, nz=false, parallel=parallel)
    elseif sim_type == "mi_nz"
        sim_mat = pw_mi_matrix(data, nz=true, parallel=parallel)
    else
        error("$sim_type is no valid similarity type.")
    end
    
    unclustered_vars = Set{Int64}(1:size(data, 2))
    clust_dict = Dict{Int64,Set{Int64}}()
    
    var_order = collect(1:size(data, 2))
    if ordering == "size"
        var_sizes = sum(data, 1)
        sort!(var_order, by=x -> var_sizes[x])
    end

    for var_A in var_order
        if var_A in unclustered_vars
            pop!(unclustered_vars, var_A)
            
            clust_members = Set(var_A)
            #rm_vars = Set{Int64}()
            for var_B in unclustered_vars
                if sim_mat[var_A, var_B] > cluster_sim_threshold
                    push!(clust_members, var_B)
                    pop!(unclustered_vars, var_B)
                end
            end
            clust_dict[var_A] = clust_members
        end
    end
    
    (sort(collect(keys(clust_dict))), clust_dict)                 
end


function clr(X::Matrix{Float64}, pseudo_count::Float64=1e-5)
    X += pseudo_count
    X = log(X ./ mapslices(geomean, X_trans, 2))
end


function discretize(x_vec::Vector{Int64}, n_bins::Int64=3; rank_method::String="dense")
    if rank_method == "dense"
        x_vec = denserank(x_vec)
    elseif rank_method == "tied"
        x_vec = tiedrank(x_vec)
    else
        error("$rank_method not a valid ranking method")
    end
    
    x_vec /= maximum(x_vec)

    step = 1.0 / n_bins

    disc_vec = map(x -> Int(floor((x-0.001) / step)), x_vec)
end
  

function discretize_nz(x_vec::Vector{Int64}, n_bins::Int64=3)
    nz_indices = findn(x_vec)
    x_vec_nz = x_vec[x_vec .> 0]
    disc_nz_vec = discretize(x_vec_nz, n_bins-1, rank_method="tied") + 1
    disc_vec = zeros(Int64, size(x_vec))
    disc_vec[nz_indices] = disc_nz_vec
    disc_vec        
end


function discretize(X::Matrix{Int64}; n_bins::Int64=3, nz::Bool=true)
    disc_fun = nz ? discretize_nz : discretize
    mapslices(x -> disc_fun(x, n_bins), X, 1)
end


function preprocess_data(data, norm::String; cluster_sim_threshold::Float64=0.0, clr_pseudo_count::Float64=1e-5, n_bins::Int64=3,
        verbose::Bool=true, skip_cols::Set{Int64}=Set{Int64}())
    if verbose
        println("Removing variables with 0 variance (or equivalently 1 level) and samples with 0 reads")
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
        data = data ./ sum(data, 2)
    elseif norm == "clr"
        data = clr(data, clr_pseudo_count)
    elseif norm == "binary"
        data = sign(data)
    elseif norm == "binned_nz"
        data = discretize(data, n_bins=n_bins, nz=true)
        unreduced_vars = size(data, 2)
        data = data[:, (map(x -> get_levels(data[:, x]), 1:size(data, 2)) .>= n_bins)[:]]
        
        if verbose
            println("\tremoved $(unreduced_vars - size(data, 2)) variables with less than $n_bins levels")
        end
            
    else
        error("$norm is no valid normalization method.")
    end
    
    data
end


function preprocess_data_default(data, test_name::String)
    default_norm_dict = Dict("mi" => "binary", "mi_nz" => "binned_nz", "fz" => "clr", "fz_nz" => "clr_nz")
    data = preprocess_data(data, default_norm_dict[test_name])
    data
end


end