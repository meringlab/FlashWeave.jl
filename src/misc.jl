module Misc

export TestResult, isdiscrete, is_zero_adjusted, is_mi_test, signed_weight, make_cum_levels!, level_map!, dict_to_adjmat

const inf_weight = 708.3964185322641

type TestResult
    stat :: Float64
    pval :: Float64
    df :: Int
    suff_power :: Bool
end


isdiscrete(test_name::String) = test_name in ["mi", "mi_nz"]
is_zero_adjusted(test_name::String) = endswith(test_name, "nz")
is_mi_test(test_name::String) = test_name in ["mi", "mi_nz"]

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