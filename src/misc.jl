const inf_weight = 708.3964185322641


function mode_string(heterogeneous, sensitive, max_k)
    het_str = heterogeneous ? "HE" : ""
    sens_str = sensitive ? "sensitive" : "fast"
    cond_str = max_k == 0 ? "univariate" : "conditional"
    "FlashWeave$het_str - $sens_str ($cond_str)"
end


function check_data(data::AbstractMatrix, meta_data::AbstractMatrix; header=nothing, meta_header=nothing)
    @assert size(data, 1) == size(meta_data, 1) "observations of data do not fit meta_data: $(size(data, 1)) vs. $(size(meta_data, 1))"

    @assert !xor(header == nothing, meta_header == nothing) "provide two headers (or none)"

    if header != nothing
        check_data(data, header)
        @assert size(meta_data, 2) == length(meta_header) "meta_header does not fit meta_data"
    end
end

function check_data(data::AbstractMatrix, header::AbstractVector; meta_mask=nothing)
    @assert size(data, 2) == length(header) "header does not fit data: $(size(data, 2)) vs. $(length(header))"
    meta_mask != nothing && @assert size(data, 2) == length(meta_mask) "meta_mask does not fit data: $(size(data, 2)) vs. $(length(meta_mask))"
end


function make_test_object(test_name::String, cond::Bool; max_k::Integer=0,
        levels::Vector{<:Integer}=Int[], cor_mat::Matrix{ContType}=zeros(ContType, 0, 0), cache_pcor::Bool=true) where ContType<:AbstractFloat
    discrete_test = isdiscrete(test_name)
    nz = is_zero_adjusted(test_name) ? Nz() : NoNz()

    if cond
        test_obj = discrete_test ? MiTestCond(levels, nz, max_k) : FzTestCond(cor_mat, Dict{String,Dict{String,ContType}}(), nz, cache_pcor)
    else
        test_obj = discrete_test ? MiTest(levels, nz) : FzTest(cor_mat, nz)
    end
    test_obj
end


function get_levels(x::Int, data::SparseMatrixCSC{ElType}) where ElType <: Integer
    unique_vals = BitSet()
    for j in nzrange(data, x)
        push!(unique_vals, data.nzval[j])
    end

    add_zero = size(data, 1) > length(nzrange(data, x)) ? 1 : 0
    length(unique_vals) + add_zero
end


function get_levels(x::Int, data::Matrix{ElType}) where ElType <: Integer
    length(unique(@view data[:, x]))
end


function get_levels(data::AbstractMatrix{ElType}) where ElType <: Integer
    map(x -> get_levels(x, data), 1:size(data, 2))
end


stop_reached(start_time::AbstractFloat, time_limit::AbstractFloat) = time_limit > 0.0 ? time() - start_time > time_limit : false


function needs_nz_view(X::Int, data::AbstractMatrix{ElType}, test_obj::AbstractTest) where ElType
    nz = is_zero_adjusted(test_obj)
    is_nz_var = iscontinuous(test_obj) || test_obj.levels[X] > 2
    nz && is_nz_var && (!issparse(data) || isa(test_obj, FzTestCond))# || isa(test_obj, MiTestCond))
end

signed_weight(test_result::TestResult, kind::AbstractString="stat") = signed_weight(test_result.stat, test_result.pval, kind)

function signed_weight(stat::Float64, pval::Float64, kind::AbstractString="stat")
    if kind == "stat"
        weight = stat
    else
        error("only 'stat' weights are currently supported")
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


function make_weights(PC_dict::OrderedDict{Int,Tuple{Float64,Float64}},
                      univar_nbrs::OrderedDict{Int,Tuple{Float64,Float64}}, weight_type::String,
                      test_name::String)
    # create weights
    nbr_dict = Dict{Int,Float64}()
    weight_kind = split(weight_type, "_")[2]

    if startswith(weight_type, "uni")
        nbr_dict = Dict([(nbr, signed_weight(univar_nbrs[nbr]..., weight_kind)) for nbr in keys(PC_dict)])
    else
        if isdiscrete(test_name)
            nbr_dict = Dict{Int,Float64}()
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


function level_map!(Zs::Tuple{Vararg{Int64,N} where N<:Int}, data::AbstractMatrix{ElType}, z::AbstractVector{<:Integer},
        cum_levels::AbstractVector{<:Integer},
    z_map_arr::AbstractVector{<:Integer}) where ElType <: Integer
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


function print_network_stats(graph::LightGraphs.Graph)
    n_nodes = nv(graph)
    n_edges = ne(graph)
    println("Current nodes/edges: $n_nodes / $n_edges")
    println("Degree stats:")
    deg = degree(graph)
    println(summarystats(deg))
    deg_median = median(deg)
    if deg_median > 20
        @warn "The network seems unusually dense (current median degree $deg_median across all nodes) which can lead to slow speed. For possible causes see <>."
    end
end


function maxweight(weight1::Float64, weight2::Float64, e1::Int, e2::Int, header=nothing)
    sign1 = sign(weight1)
    sign2 = sign(weight2)

    if isnan(weight1)
        return weight2
    elseif isnan(weight2)
        return weight1
    else
        if sign1 * sign2 < 0
            e1w, e2w = header != nothing ? (e1, e2) : (header[e1], header[e2])
            @warn "Opposite signs for edge $e1w <-> $e2w detected. Arbitarily choosing one."
            return weight1
        else
            return max(abs(weight1), abs(weight2)) * sign1
        end
    end
end


order_pair(x1, x2) = x1 >= x2 ? (x1, x2) : (x2, x1)


function SimpleWeightedGraph_nodemax(i::AbstractVector{T}, j::AbstractVector{T}, v::AbstractVector{U}; combine = +, m=max(maximum(i), maximum(j))) where T<:Integer where U<:Real
    s = sparse(vcat(i,j), vcat(j,i), vcat(v,v), m, m, combine)
    SimpleWeightedGraph{T, U}(s)
end


function make_symmetric_graph(weights_dict::Dict{Int,Dict{Int,Float64}}, edge_rule::String;
    edge_merge_fun=maxweight, max_var::Int=-1, header=nothing)

    if max_var < 0
        max_val_key = maximum(map(x -> !isempty(x) ? maximum(keys(x)) : 0, values(weights_dict)))
        max_key_key = maximum(keys(weights_dict))
        max_var = max(max_key_key, max_val_key)
    end

    srcs = Int[]
    dsts = Int[]
    ws = Float64[]

    prev_edges = Set{Tuple{Int,Int}}()
    for node1 in keys(weights_dict)
        for node2 in keys(weights_dict[node1])
            e = order_pair(node1, node2)

            if !(e in prev_edges)
                weight = weights_dict[node1][node2]
                rev_weight = get(weights_dict[node2], node1, NaN64)

                sym_weight = edge_merge_fun(weight, rev_weight, node1, node2, header)
                push!(srcs, e[1])
                push!(dsts, e[2])
                push!(ws, sym_weight)
                push!(prev_edges, e)
            end
        end
    end

    SimpleWeightedGraph_nodemax(srcs, dsts, ws; m=max_var)
end


function iter_apply_sparse_rows!(X::Int, Y::Int, data::SparseMatrixCSC{ElType},
        red_fun, red_obj, x_nzadj=false, y_nzadj=false) where ElType <: Real
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
            x_entry = zero(ElType)
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
            y_entry = zero(ElType)
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
