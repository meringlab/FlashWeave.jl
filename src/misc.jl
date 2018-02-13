module Misc

using LightGraphs
using StatsBase
using Combinatorics
using DataStructures
using JLD2
using FileIO
using MetaGraphs

using FlashWeave.Types

export make_test_object, needs_nz_view, get_levels, stop_reached, signed_weight,
       workers_all_local, make_cum_levels!, make_cum_levels, level_map!,
       print_network_stats, maxweight, make_symmetric_graph, map_edge_keys,
       pw_unistat_matrix, dict_to_adjmat, make_weights, iter_apply_sparse_rows!,
       make_chunks, work_chunker

const inf_weight = 708.3964185322641


function make_test_object{ContType<:AbstractFloat}(test_name::String, cond::Bool; max_k::Integer=0,
        levels::Vector{<:Integer}=Int[], cor_mat::Matrix{ContType}=zeros(ContType, 0, 0), cache_pcor::Bool=true)
    discrete_test = isdiscrete(test_name)
    nz = is_zero_adjusted(test_name) ? Nz() : NoNz()

    if cond
        test_obj = discrete_test ? MiTestCond(levels, nz, max_k) : FzTestCond(cor_mat, Dict{String,Dict{String,ContType}}(), nz, cache_pcor)
    else
        test_obj = discrete_test ? MiTest(levels, nz) : FzTest(cor_mat, nz)
    end
    test_obj
end


function get_levels{ElType <: Integer}(col_vec::SparseVector{ElType,Int})
    levels = length(unique(nonzeros(col_vec)))
    add_zero = col_vec.n > length(col_vec.nzind) ? one(ElType) : zero(ElType)
    levels + add_zero
end


function get_levels{ElType <: Integer}(col_vec::AbstractVector{ElType})
    length(unique(col_vec))
end

function get_levels{ElType <: Integer}(data::AbstractMatrix{ElType})
    map(x -> get_levels(data[:, x]), 1:size(data, 2))
end


stop_reached(start_time::AbstractFloat, time_limit::AbstractFloat) = time_limit > 0.0 ? time() - start_time > time_limit : false

function needs_nz_view{ElType}(X::Int, data::AbstractMatrix{ElType}, test_obj::AbstractTest)
    nz = is_zero_adjusted(test_obj)
    is_nz_var = iscontinuous(test_obj) || test_obj.levels[X] > 2
    nz && is_nz_var && (!issparse(data) || isa(test_obj, FzTestCond))# || isa(test_obj, MiTestCond))
end

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


function make_weights(PC_dict::OrderedDict{Int,Tuple{Float64,Float64}}, univar_nbrs::OrderedDict{Int,Tuple{Float64,Float64}}, weight_type::String, test_name::String)
    # create weights
    nbr_dict = Dict{Int,Float64}()
    weight_kind = String(split(weight_type, "_")[2])

    if startswith(weight_type, "uni")
        nbr_dict = Dict([(nbr, signed_weight(univar_nbrs[nbr]..., weight_kind)) for nbr in keys(PC_dict)])
    else
        if startswith(test_name, "mi")
            nbr_dict = Dict{Int, Float64}()
            for nbr in keys(PC_dict)
                edge_sign = sign(univar_nbrs[nbr][1])
                nbr_dict[nbr] = edge_sign * abs(signed_weight(PC_dict[nbr]..., weight_kind))
            end
        else
            nbr_dict = Dict([(nbr, signed_weight(PC_dict[nbr]..., weight_kind)) for nbr in keys(PC_dict)])
        end
    end

    nbr_dict
end


function make_cum_levels!{ElType <: Integer}(cum_levels::AbstractVector{ElType}, Zs::Tuple{Vararg{Int64,N} where N<:Int}, levels::AbstractVector{ElType})

    cum_levels[1] = 1

    @inbounds for j in 2:length(Zs)
        Z_var = Zs[j]
        levels_z = levels[Z_var]
        cum_levels[j] = cum_levels[j - 1] * levels_z
    end
end

function make_cum_levels{ElType <: Integer}(Zs::Tuple{Vararg{Int64,N} where N<:Int}, levels::AbstractVector{ElType})
    cum_levels = zeros(Int, length(Zs))
    make_cum_levels!(cum_levels, Zs, levels)
    cum_levels
end


function level_map!{ElType <: Integer}(Zs::Tuple{Vararg{Int64,N} where N<:Int}, data::AbstractMatrix{ElType}, z::AbstractVector{<:Integer},
        cum_levels::AbstractVector{<:Integer},
    z_map_arr::AbstractVector{<:Integer})
    fill!(z_map_arr, -1)
    levels_z = zero(ElType)

    @inbounds for i in 1:size(data, 1)
        gfp_map = one(ElType)
        for (j, Z_var) in enumerate(Zs)
            gfp_map += data[i, Z_var] * cum_levels[j]
        end

        level_val = z_map_arr[gfp_map]
        if level_val != -1
            z[i] = level_val
        else
            z_map_arr[gfp_map] = levels_z
            z[i] = levels_z
            levels_z += one(ElType)
        end
    end

    levels_z
end

function dict_to_graph{T}(graph_dict::Dict{T,Dict{T,Float64}})
    G = Graph(maximum(keys(graph_dict)))
    for key in keys(graph_dict)
        @assert key <= nv(G) "nodes are missing from graph_dict"
        for nbr in keys(graph_dict[key])
            @assert nbr <= nv(G) "nodes are missing from graph_dict"
            add_edge!(G, var_A, var_B)
        end
    end
    G
end


function dict_to_metagraph{T}(graph_dict::Dict{T,Dict{T,Float64}})
    Gm = MetaGraph(length(graph_dict))
    for (i, (key, nbr_dict)) in enumerate(graph_dict)
        @assert key <= nv(G) "nodes are missing from graph_dict"
        for (nbr, weight) in nbr_dict
            if !has_edge(G, key, nbr)
                @assert nbr <= nv(G) "nodes are missing from graph_dict"
                add_edge!(G, key, nbr)
                set_prop!(G, key, nbr, :weight, nbr_dict[nbr])
            end
        end
    end
    G
end


function neighbor_distances(G1, G2, G2_sps=zeros(Float64, 0, 0))
    """Shortest path distances in G2, for each edge present in G1"""
    if isempty(G2_sps)
        G2_sps = convert(Matrix{Float64}, floyd_warshall_shortest_paths(G2).dists)
    end
    G2_sps[G2_sps .> nv(G2)] = NaN64
    nbr_dists = [G2_sps[e.src, e.dst] for e in edges(G1)]
    nbr_dists
end

function jaccard_similarity(graph_dict1::Dict, graph_dict2::Dict)
    G1 = dict_to_graph(graph_dict1)
    G2 = dict_to_graph(graph_dict2)
    jaccard_similarity(G1, G2)
end

function jaccard_similarity(G1::LightGraphs.Graph, G2::LightGraphs.Graph)
    edge_set1 = Set(map(Tuple, edges(G1)))
    edge_set2 = Set(map(Tuple, edges(G2)))
    length(intersect(edge_set1, edge_set2)) / length(union(edge_set1, edge_set2))
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
            return maximum(abs.([weight1, weight2])) * sign1
        end
    end
end


function make_graph_symmetric(weights_dict::Dict{Int,Dict{Int,Float64}}, edge_rule::String, edge_merge_fun=maxweight)
    checked_G = Graph(maximum(keys(weights_dict)))
    graph_dict = Dict{Int,Dict{Int,Float64}}([(target_var, Dict{Int,Float64}()) for target_var in keys(weights_dict)])

    for node1 in keys(weights_dict)
        for node2 in keys(weights_dict[node1])

            if !has_edge(checked_G, node1, node2)
                add_edge!(checked_G, node1, node2)
                weight = weights_dict[node1][node2]

                # if only one direction is present and "AND" rule is specified, skip this edge
                if edge_rule == "AND" && !haskey(weights_dict[node2], node1)
                    continue
                end

                prev_weight = haskey(weights_dict[node2], node1) ? weights_dict[node2][node1] : NaN64

                weight = edge_merge_fun(weight, prev_weight)

                graph_dict[node1][node2] = weight
                graph_dict[node2][node1] = weight
            end
        end
    end

    graph_dict
end

function make_symmetric_graph(weights_dict::Dict{Int,Dict{Int,Float64}}, edge_rule::String, edge_merge_fun=maxweight)
    G = MetaGraph(maximum(keys(weights_dict)))

    for node1 in keys(weights_dict)
        for node2 in keys(weights_dict[node1])

            if !has_edge(G, node1, node2)
                # if only one direction is present and "AND" rule is specified, skip this edge
                if edge_rule == "AND" && !haskey(weights_dict[node2], node1)
                    continue
                end

                e = Edge(node1, node2)
                add_edge!(G, e)

                weight = weights_dict[node1][node2]
                rev_weight = haskey(weights_dict[node2], node1) ? weights_dict[node2][node1] : NaN64

                weight = edge_merge_fun(weight, rev_weight)
                if edge_rule == "OR"
                    if haskey(weights_dict[node2], node1)
                        edge_dir = '='
                    elseif node1 < node2
                        edge_dir = '>'
                    else
                        edge_dir = '<'
                    end
                    set_props!(G, e, Dict(:weight=>weight, :dir=>edge_dir))
                else
                    set_prop!(G, e, :weight, weight)
                end
            end
        end
    end

    G
end


function map_edge_keys(nbr_dict::Dict{Int,T}, key_map_dict::Dict{Int,Int}) where T <: Associative{Int}
    new_nbr_dict = similar(nbr_dict)

    for (key, sub_dict) in nbr_dict
        var_key = key_map_dict[key]
        new_sub_dict = similar(sub_dict)

        for (sub_key, sub_val) in sub_dict
            if haskey(key_map_dict, sub_key)
                var_sub_key = key_map_dict[sub_key]
                new_sub_dict[var_sub_key] = sub_val
            end
        end

        new_nbr_dict[var_key] = new_sub_dict
    end
    new_nbr_dict
end

function metagraph_to_adjmat(G::MetaGraph, header::AbstractVector{String})
    n_vars = length(header)
    adj_mat = zeros(Float64, (n_vars, n_vars))
    checked_nodes = Set{Int}()
    for node_index in 1:n_vars
        for nbr in neighbors(G, node_index)
            if !(nbr in checked_nodes)
                weight = get_prop(G, node_index, nbr, :weight)
                adj_mat[node_index, nbr] = weight
                adj_mat[nbr, node_index] = weight
            end
        end
    end
    adj_mat_final = vcat(reshape(header, 1, size(adj_mat, 2)), adj_mat)
    adj_mat_final = hcat(reshape(["", header...], size(adj_mat_final, 2) + 1, 1), adj_mat_final)
    adj_mat_final
end

function dict_to_adjmat(graph_dict::Dict{Int,Dict{Int,Float64}}, header::AbstractVector{String})
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


function translate_hiton_state(state::HitonState{T1}, trans_dict::Dict{T1,T2}) where {T1,T2}
    trans_state_results = OrderedDict{T2,Tuple{Float64,Float64}}([(trans_dict[key], val) for (key, val) in state.state_results])
    trans_inter_results = OrderedDict{T2,Tuple{Float64,Float64}}([(trans_dict[key], val) for (key, val) in state.inter_results])
    trans_unchecked_vars = T2[trans_dict[x] for x in state.unchecked_vars]
    trans_state_rejections = Dict{String,Tuple{Tuple,TestResult}}([(trans_dict[key], (Tuple(map(x -> trans_dict[x], Zs)), test_res)) for (key, (Zs, test_res)) in state.state_rejections])
    HitonState{T2}(state.phase, trans_state_results, trans_inter_results, trans_unchecked_vars, trans_state_rejections)
end

function translate_graph_dict(graph_dict::Dict{T1,Dict{T1,S}}, trans_dict::Dict{T1,T2}) where {T1,T2,S}
    new_graph_dict = Dict{T2,Dict{T2,S}}()
    for (key, nbr_dict) in graph_dict
        new_graph_dict[trans_dict[key]] = Dict{T2,S}([(trans_dict[key], val) for (key, val) in nbr_dict])
    end
    new_graph_dict
end

function translate_results(results::LGLResult{T1}, trans_dict::Dict{T1,T2}) where {T1,T2}
    trans_graph = translate_graph_dict(results.graph, trans_dict)
    trans_rejections = translate_graph_dict(results.rejections, trans_dict)
    for (key, nbr_dict) in trans_rejections
        for (nbr, (Zs, test_res)) in nbr_dict
            trans_rejections[key][nbr] = (Tuple(map(x -> trans_dict[x], Zs)), test_res)
        end
    end

    trans_unfinished_states = Dict{T2,HitonState{T2}}([(trans_dict[key], translate_hiton_state(val, trans_dict)) for (key, val) in results.unfinished_states])
    LGLResult{T2}(trans_graph, trans_rejections, trans_unfinished_states)
end

function translate_results(results::LGLResult{T1}, header::Vector{T2}) where {T1,T2}
    trans_dict = Dict{T1,T2}(enumerate(header))
    translate_results(results, trans_dict)
end

# hacky/slow functions to get around JLD2 problems in saving/loading LGLResult objects
function save_lglresult(out_path::String, net_obj::LGLResult)
    weight_dict = Dict{Tuple{Int,Int},Float64}()
    for (e, props) in net_obj.graph.eprops
        e_tup = (e.src, e.dst)
        weight_dict[e_tup] = props[:weight]
    end

    save(out_path, Dict("weight_dict"=>weight_dict, "rejections"=>Int,
            "unfinished_states"=>Int, "max_node"=>nv(net_obj.graph)))
end

function load_lglresult(in_path)
    d = load(in_path)
    weight_dict = d["weight_dict"]
    G = MetaGraph(d["max_node"])
    eprop_dict = Dict{Edge, Dict{Symbol, Any}}()

    for ((src, dst), weight) in weight_dict
        add_edge!(G, src, dst)
        set_prop!(G, src, dst, :weight, weight)
    end
    LGLResult(G, Dict{Int, Dict{Int, Tuple{Tuple,TestResult}}}(), Dict{Int, HitonState}())
end


function save_network(results::LGLResult, out_path::String; meta_dict::Dict=Dict(), fmt::String="auto",
        header::Vector{String}=String[])
    """Currently unmaintained"""
    if fmt == "auto"
        fmt = split(out_path, ".")[end]
    end

    if fmt == "jld"
        save(out_path, "results", results, "meta_data", meta_dict)
    else
        base_path = splitext(out_path)[1]
        meta_path = join([base_path, "_meta_data.txt"])

        if !isempty(meta_dict)
            open(meta_path, "w") do meta_f
                for key in keys(meta_dict)
                    write(meta_f, string(key), " : ", string(meta_dict[key]))
                end
            end
        end

        if fmt == "adj"
            if isempty(header)
                error("fmt \"adj\" can only be used if header is provided")
            end
            adj_mat = dict_to_adjmat(results.graph, header)
            writedlm(out_path, adj_mat, '\t')
        else
            error("fmt \"$fmt\" is not a valid output format.")
        end
    end
end



function iter_apply_sparse_rows!{ElType <: Real}(X::Int, Y::Int, data::SparseMatrixCSC{ElType},
        red_fun, red_obj, x_nzadj=false, y_nzadj=false)
    n_rows, n_cols = size(data)
    num_out_of_bounds = 0
    row_inds = rowvals(data)
    vals = nonzeros(data)

    x_i = data.colptr[X]
    x_row_ind = row_inds[x_i]
    x_val = vals[x_i]

    if X != n_cols
        x_bound = data.colptr[X + 1]
    else
        x_bound = nnz(data)
    end

    if x_i == x_bound
        if x_nzadj
            return
        else
            num_out_of_bounds += 1
        end
    end

    y_i = data.colptr[Y]
    y_row_ind = row_inds[y_i]
    y_val = vals[y_i]

    if Y != n_cols
        y_bound = data.colptr[Y + 1]
    else
        y_bound = nnz(data) + 1
    end

    if y_i == y_bound
        if y_nzadj
            return
        else
            num_out_of_bounds += 1
        end
    end

    min_row_ind = min(x_row_ind, y_row_ind)

    while true
        skip_row = false
        if x_row_ind == min_row_ind
            x_entry = x_val
            x_i += 1

            if x_i < x_bound
                x_row_ind = row_inds[x_i]
                x_val = vals[x_i]
            else
                num_out_of_bounds += 1
                x_row_ind = n_rows + 1
            end
        else
            x_entry = zero(eltype(data))
            skip_row = x_nzadj
        end

        if y_row_ind == min_row_ind
            y_entry = y_val
            y_i += 1

            if y_i < y_bound
                y_row_ind = row_inds[y_i]
                y_val = vals[y_i]
            else
                num_out_of_bounds += 1
                y_row_ind = n_rows + 1
            end
        else
            y_entry = zero(eltype(data))
            skip_row = y_nzadj
        end

        min_row_ind = min(x_row_ind, y_row_ind)

        if !skip_row
            red_fun(red_obj, x_entry, y_entry)
        end

        if num_out_of_bounds >= 2
            break
        end
    end
end

make_chunks(a::AbstractVector, chunk_size, offset) = (i:min(maximum(a), i + chunk_size - 1) for i in offset+1:chunk_size:maximum(a))
work_chunker(n_vars, chunk_size=1000) = ((X, Y_slice) for X in 1:n_vars-1 for Y_slice in make_chunks(X+1:n_vars, chunk_size, X))

end
