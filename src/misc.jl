module Misc

export HitonState, TestResult, stop_reached, isdiscrete, is_zero_adjusted, is_mi_test, signed_weight, workers_all_local, make_cum_levels!, level_map!, dict_to_adjmat, make_weights

const inf_weight = 708.3964185322641

type HitonState
    phase :: String
    state_results :: Dict{Int,Tuple{Float64,Float64}}
    unchecked_vars :: Vector{Int}
end

type TestResult
    stat :: Float64
    pval :: Float64
    df :: Int
    suff_power :: Bool
end

stop_reached(start_time::Float64, time_limit::Float64) = time_limit > 0.0 ? time() - start_time > time_limit : false

isdiscrete(test_name::String) = test_name in ["mi", "mi_nz"]
is_zero_adjusted(test_name::String) = endswith(test_name, "nz")
is_mi_test(test_name::String) = test_name in ["mi", "mi_nz"]

signed_weight(test_result::TestResult, kind::String="logpval") = signed_weight(test_result.stat, test_result.pval, kind)


function workers_all_local()
    local_host = gethostname()
    workers_local = true
    
    for worker_id in workers()
        worker_host = remotecall_fetch(()->gethostname(), worker_id)
        if worker_host != local_host
            workers_local = false
            break
        end
    end
    workers_local
end

function signed_weight(stat::Float64, pval::Float64, kind::String="logpval")
    if kind == "stat"
        weight = stat
    elseif endswith(kind, "pval")
        sign_factor = stat < 0.0 ? -1.0 : 1.0
    
        if kind == "logpval"
            weight = -log(pval)
            weight = isinf(weight) ? inf_weight : weight
        else
            weight = pval
        end
        weight *= sign_factor
    end
    weight
end


function make_weights(PC_dict, univar_nbrs, weight_type)
    # create weights
    nbr_dict = Dict{Union{Int, String},Float64}()
    weight_kind = String(split(weight_type, "_")[2])
    if startswith(weight_type, "uni")
        nbr_dict = Dict([(nbr, signed_weight(univar_nbrs[nbr]..., weight_kind)) for nbr in keys(PC_dict)])
    else
        nbr_dict = Dict([(nbr, signed_weight(PC_dict[nbr]..., weight_kind)) for nbr in keys(PC_dict)])
    end
    
    nbr_dict
end


function make_cum_levels!(cum_levels::Vector{Int}, Zs::Vector{Int}, levels::Vector{Int})
    cum_levels[1] = 1

    for j in 2:length(Zs)
        Z_var = Zs[j]
        levels_z = levels[Z_var]
        cum_levels[j] = cum_levels[j - 1] * levels_z
    end
end


function level_map!(Zs::Vector{Int}, data::Union{SubArray,Matrix{Int64}}, z::Vector{Int}, cum_levels::Vector{Int},
    z_map_arr::Vector{Int})
    fill!(z_map_arr, 0)
    levels_z = 1
    
    for i in 1:size(data, 1)
        gfp_map = 1
        for (j, Z_var) in enumerate(Zs)
            gfp_map += data[i, Z_var] * cum_levels[j]
        end
        
        level_val = z_map_arr[gfp_map]
        if level_val != 0
            z[i] = level_val
        else
            z_map_arr[gfp_map] = levels_z
            z[i] = levels_z
            levels_z += 1   
        end
    end
    
    levels_z
end


function dict_to_adjmat(graph_dict::Dict{Union{Int64,String},Dict{Union{Int64,String},Float64}})
    n_nodes = length(graph_dict)
    adj_mat = zeros(Float64, (n_nodes, n_nodes))
    header = sort(collect(keys(graph_dict)))
    header_map = Dict(zip(header, 1:n_nodes))
    
    for node in keys(graph_dict)
        node_index = header_map[node]
        nbr_dict = graph_dict[node]
        
        for nbr in keys(nbr_dict)
            nbr_index = header_map[nbr]
            weight = nbr_dict[nbr]
            
            adj_mat[node_index, nbr_index] = weight
            adj_mat[nbr_index, node_index] = weight
        end
    end    
    
    adj_mat = vcat(reshape(header, 1, size(adj_mat, 2)), adj_mat)
    adj_mat = hcat(reshape(["", header...], size(adj_mat, 2) + 1, 1), adj_mat)
    adj_mat
end

end