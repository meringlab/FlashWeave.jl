module Misc

using LightGraphs
using StatsBase

export HitonState, TestResult, IndexPair, get_levels, min_sec_indices!, stop_reached, isdiscrete, is_zero_adjusted, is_mi_test, signed_weight, workers_all_local, make_cum_levels!, level_map!, print_network_stats, maxweight, make_graph_symmetric, dict_to_adjmat, make_weights

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

type IndexPair
    min_ind :: Int64
    sec_ind :: Int64
end



function get_levels(col_vec::SparseVector{Int64,Int64})
    levels = length(unique(nonzeros(col_vec)))
    add_zero = col_vec.n > length(col_vec.nzind) ? 1 : 0
    levels + add_zero
end


function get_levels(col_vec::Vector{Int64})
    length(unique(col_vec))
end


function min_sec_indices!(ind_pair::IndexPair, index_vec::Vector{Int64})
    min_ind = 0
    sec_ind = 0
    
    for ind in index_vec
        if min_ind == 0 || ind < min_ind
            sec_ind = min_ind
            min_ind = ind
        elseif sec_ind == 0 || ind < sec_ind
            sec_ind = ind
        end
    end
    ind_pair.min_ind = min_ind
    ind_pair.sec_ind = sec_ind
end


stop_reached(start_time::Float64, time_limit::Float64) = time_limit > 0.0 ? time() - start_time > time_limit : false

isdiscrete(test_name::String) = test_name in ["mi", "mi_nz", "mi_expdz"]
is_zero_adjusted(test_name::String) = endswith(test_name, "nz")
is_mi_test(test_name::String) = test_name in ["mi", "mi_nz", "mi_expdz"]

signed_weight(test_result::TestResult, kind::String="logpval") = signed_weight(test_result.stat, test_result.pval, kind)

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


#function colsize(data::SparseMatrixCSC, col::Int)
#    col_end_term = col < size(data, 2) ? data.colptr[col + 1] : nnz(data) + 1
#    col_end_term - data.colptr[col]
#end


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


function make_weights(PC_dict, univar_nbrs, weight_type)
    # create weights
    nbr_dict = Dict{Int,Float64}()
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
    fill!(z_map_arr, -1)
    levels_z = 0
    
    for i in 1:size(data, 1)
        gfp_map = 1
        for (j, Z_var) in enumerate(Zs)
            gfp_map += data[i, Z_var] * cum_levels[j]
        end
        
        level_val = z_map_arr[gfp_map]
        if level_val != -1
            z[i] = level_val
        else
            z_map_arr[gfp_map] = levels_z
            z[i] = levels_z
            levels_z += 1   
        end
    end
    
    levels_z
end


function print_network_stats(graph::LightGraphs.Graph)
    n_nodes = nv(graph)
    n_edges = ne(graph)
    println("Current nodes/edges: $n_nodes / $n_edges")
    println("Degree stats:")
    deg = degree(graph)
    println(summarystats(deg))
    deg_median = median(deg)
    if deg_median > 20
        warn("The network seems unusually dense (current median degree $deg_median across all nodes) which can lead to slow speed. For possible causes see <>.")
    end
end


function maxweight(weight1::Float64, weight2::Float64)
    sign1 = sign(weight1) 
    sign2 = sign(weight2)
    
    if isnan(weight1)
        return weight2
    elseif isnan(weight2)
        return weight1
    else
        if sign1 * sign2 < 0
            warn("Opposite signs for the same edge detected. Arbitarily choosing one.")
            return weight1
        else
            return maximum(abs([weight1, weight2])) * sign1
        end
    end
end

    
function make_graph_symmetric(weights_dict::Dict{Int64,Dict{Int64,Float64}}, edge_merge_fun=maxweight)
    checked_G = Graph(maximum(keys(weights_dict)))
    graph_dict = Dict{Int64,Dict{Int64,Float64}}([(target_var, Dict{Int64,Float64}()) for target_var in keys(weights_dict)])
    
    for node1 in keys(weights_dict)
        for node2 in keys(weights_dict[node1])
            
            if !has_edge(checked_G, node1, node2)
                add_edge!(checked_G, node1, node2)
                weight = weights_dict[node1][node2]

                prev_weight = haskey(weights_dict[node2], node1) ? weights_dict[node2][node1] : NaN64

                #if prev_weight != 0.0 && !isnan(weight)
                weight = edge_merge_fun(weight, prev_weight)
                #end

                graph_dict[node1][node2] = weight
                graph_dict[node2][node1] = weight
            end
        end
    end
    
    graph_dict
end
    

#function dict_to_graph(graph_dict::Dict{Int64,Dict{Int64,Float64}}, edge_merge_fun=maxweight)
#    max_key = maximum(keys(graph_dict))
#    adj_mat = zeros(Float64, max_key, max_key)
#    #graph = Graph(maximum(keys(graph_dict)))
#    
#    for node1 in keys(graph_dict)
#        for node2 in keys(graph_dict[node1])
#            weight = graph_dict[node1][node2]
#            prev_weight = adj_mat[node1, node2]
#            
#            if prev_weight != 0.0 && !isnan(weight)
#                weight = edge_merge_fun([weight, prev_weight])
#            end
#            
#            adj_mat[node1, node2] = weight
#            adj_mat[node2, node1] = weight
#            
#        end
#    end
#    
#    Graph(adj_mat)
#end 


function dict_to_adjmat(graph_dict::Dict{Union{Int64,String},Dict{Union{Int64,String},Float64}}, header::Vector{String})
    #n_nodes = length(graph_dict)
    n_vars = length(header)
    adj_mat = zeros(Float64, (n_vars, n_vars))
    #header = sort(collect(keys(graph_dict)))
    #header_map = Dict(zip(header, 1:length(header)))
    
    for node_index in keys(graph_dict)
        #node_index = #header_map[node]
        nbr_dict = graph_dict[node_index]
        
        for nbr_index in keys(nbr_dict)
            #nbr_index = header_map[nbr]
            weight = nbr_dict[nbr_index]
            
            adj_mat[node_index, nbr_index] = weight
            adj_mat[nbr_index, node_index] = weight
        end
    end    
    
    adj_mat = vcat(reshape(header, 1, size(adj_mat, 2)), adj_mat)
    adj_mat = hcat(reshape(["", header...], size(adj_mat, 2) + 1, 1), adj_mat)
    adj_mat
end

end