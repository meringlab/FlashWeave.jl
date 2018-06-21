module Interleaved

using LightGraphs
using SimpleWeightedGraphs
using DataStructures

using FlashWeave.Misc
using FlashWeave.Types
using FlashWeave.Hiton
using FlashWeave.StackChannels

export interleaved_backend

function interleaved_worker(data::AbstractMatrix{ElType}, levels, cor_mat, edge_rule::String, nonsparse_cond::Bool,
     shared_job_q::RemoteChannel, shared_result_q::RemoteChannel, GLL_args::Dict{Symbol,Any}) where {ElType<:Real}

    if nonsparse_cond
        data = full(data)
    end

    const converged = false

    while true
        try
            target_var, univar_nbrs, prev_state, skip_nbrs = take!(shared_job_q)
            # if kill signal
            if target_var == -1
                put!(shared_result_q, (0, myid()))
                return
            end

            if prev_state.phase == 'C'
                converged = true
            elseif converged
                prev_state = HitonState('C', prev_state.state_results, prev_state.inter_results, prev_state.unchecked_vars,
                                        prev_state.state_rejections)
            end

            if edge_rule == "AND"
                blacklist = skip_nbrs
                whitelist = Set{Int}()
            else
                blacklist = Set{Int}()
                whitelist = skip_nbrs
            end
            #println("computing neighbors")
            #@code_warntype si_HITON_PC(target_var, data, levels, cor_mat; univar_nbrs=univar_nbrs, prev_state=prev_state, blacklist=blacklist, whitelist=whitelist, GLL_args...)
            #println(length(univar_nbrs), " ", size(data), " ", length(levels))
            #println(typeof(levels), " ", typeof(cor_mat))
            #println("worker: whiteblack: ", length(whitelist), " ", length(blacklist))
            #println("worker: whitelist pre: ", whitelist)
            #println("worker: target_var: ", target_var)
            nbr_state = si_HITON_PC(target_var, data, levels, cor_mat; univar_nbrs=univar_nbrs,
                                    prev_state=prev_state, blacklist=blacklist, whitelist=whitelist, GLL_args...)

            #nbr_state = si_HITON_PC(target_var, data, levels, cor_mat; univar_nbrs=univar_nbrs,
            #                        prev_state=prev_state, GLL_args...)
            #nbr_state = si_HITON_PC(target_var, data, levels, cor_mat; univar_nbrs=univar_nbrs, GLL_args...)
            #println("delivering neighbors")
            put!(shared_result_q, (target_var, nbr_state))
        catch exc
            println("Exception occurred! ", exc)
            println(catch_stacktrace())
            #put!(shared_result_q, (target_var, exc))
            #throw(exc)
        end

    end
end


function interleaved_backend(target_vars::AbstractVector{Int}, data::AbstractMatrix{ElType},
        all_univar_nbrs::Dict{Int,OrderedDict{Int,Tuple{Float64,Float64}}}, levels::Vector{DiscType}, cor_mat::Matrix{ContType}, GLL_args::Dict{Symbol,Any};
        update_interval::Real=30.0, output_folder::String="", output_interval::Real=update_interval*10,
        temp_output_type::String="single",
        convergence_threshold::AbstractFloat=0.01,
        conv_check_start::AbstractFloat=0.1, conv_time_step::AbstractFloat=0.1, parallel::String="multi_il",
        edge_rule::String="OR", edge_merge_fun=maxweight, nonsparse_cond::Bool=false, verbose::Bool=true, workers_local::Bool=true,
        feed_forward::Bool=true) where {ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}

    test_name = GLL_args[:test_name]
    weight_type = GLL_args[:weight_type]
    jobs_total = length(target_vars)

    if startswith(parallel, "multi") || startswith(parallel, "threads")
        n_workers = nprocs() - 1
        job_q_buff_size = n_workers * 2
        @assert n_workers > 0 "Need to add workers for parallel processing."
    elseif startswith(parallel, "single")
        n_workers = 1
        job_q_buff_size = 1
    else
        error("$parallel not a valid execution mode.")
    end

    shared_job_q = RemoteChannel(() -> StackChannel{Tuple}(size(data, 2) * 2), 1)
    shared_result_q = RemoteChannel(() -> Channel{Tuple}(size(data, 2)), 1)

    # initialize jobs
    queued_jobs = 0
    waiting_vars = Stack(Int)
    for (i, target_var) in enumerate(reverse(target_vars))
        job = (target_var, all_univar_nbrs[target_var], HitonState{Int}('S', OrderedDict(), OrderedDict(),
                                                                        [], Dict()), Set{Int}())

        if i < jobs_total - n_workers
            push!(waiting_vars, target_var)
        else
            put!(shared_job_q, job)
            queued_jobs += 1
        end
    end


    if verbose
        println("Starting workers and sending data..")
        tic()
    end

    if !all([fetch(@spawnat workers()[1] isdefined(remote_obj)) for remote_obj in [:remote_data, :remote_test_obj]])
        remote_data = data
        remote_cor_mat = cor_mat
        remote_levels = levels
    end

    worker_returns = [@spawn interleaved_worker(remote_data, remote_levels, remote_cor_mat, edge_rule, nonsparse_cond,
                                                shared_job_q, shared_result_q, GLL_args) for x in 1:n_workers]

    if verbose
        println("Done. Starting inference..")
        toc()
    end

    remaining_jobs = jobs_total
    n_vars = size(data, 2)
    graph_dict = Dict{Int, HitonState{Int}}()

    # this graph is just used for efficiently keeping track of graph stats during the run
    graph = Graph(n_vars)

    if edge_rule == "AND"
        blacklist_graph = Graph(n_vars)
    end

    if !isempty(output_folder)
        if temp_output_type == "single"
            temp_out_path = joinpath(output_folder, "latest_network.edgelist")
        end
        output_graph = SimpleWeightedGraph(n_vars)

        !isdir(output_folder) && mkdir(output_folder)
    end

    edge_set = Set{Tuple{Int,Int}}()
    kill_signals_sent = 0
    start_time = time()
    last_update_time = start_time
    last_output_time = start_time
    check_convergence = false
    converged = false

    while remaining_jobs > 0
        target_var, nbr_result = take!(shared_result_q)
        queued_jobs -= 1
        if isa(nbr_result, HitonState{Int})
            curr_state = nbr_result

            # node has not yet finished computing
            if curr_state.phase != 'F' && curr_state.phase != 'C'
                if converged
                    curr_state = HitonState('C', curr_state.state_results, curr_state.inter_results, curr_state.unchecked_vars, curr_state.state_rejections)
                end

                if feed_forward
                    skip_nbrs = edge_rule == "AND" ? Set(neighbors(blacklist_graph, target_var)) : Set(neighbors(graph, target_var))
                else
                    skip_nbrs= Set{Int}()
                end

                job = (target_var, all_univar_nbrs[target_var], curr_state, skip_nbrs)
                put!(shared_job_q, job)
                queued_jobs += 1

            # node is complete
            else
                #println("MASTER: node complete, updating graph")
                graph_dict[target_var] = curr_state

                for nbr in keys(curr_state.state_results)
                    add_edge!(graph, target_var, nbr)
                end

                # update output graph if requested
                if !isempty(output_folder)
                    for nbr in keys(curr_state.state_results)
                        weight = make_single_weight(curr_state.state_results[nbr]..., all_univar_nbrs[target_var][nbr]..., weight_type, test_name)

                        rev_weight = has_edge(output_graph, target_var, nbr) ? output_graph.weights[target_var, nbr] : NaN64
                        #add_edge!(output_graph, target_var, nbr, edge_merge_fun(weight, rev_weight))
                        sym_weight = edge_merge_fun(weight, rev_weight)
                        output_graph.weights[target_var, nbr] = sym_weight
                        output_graph.weights[nbr, target_var] = sym_weight
                    end
                end

                if feed_forward && edge_rule == "AND"
                    for a_var in target_vars
                        if !haskey(curr_state.state_results, a_var)
                            add_edge!(blacklist_graph, target_var, a_var)
                        end
                    end
                end

                remaining_jobs -= 1

                # kill workers if not needed anymore
                if remaining_jobs < n_workers
                    kill_signal = (-1, Dict{Int,Tuple{Float64,Float64}}(), HitonState{Int}('S', OrderedDict(), OrderedDict(), [], Dict()), Set{Int}())
                    put!(shared_job_q, kill_signal)
                    kill_signals_sent += 1
                end
            end
        elseif isa(nbr_result, Int)
            if !workers_local
                rmprocs(nbr_result)
            end
        else
            println(nbr_result)
            throw(nbr_result)
        end

        if !isempty(waiting_vars) && queued_jobs < job_q_buff_size
            for i in 1:job_q_buff_size - queued_jobs
                next_var = pop!(waiting_vars)

                if feed_forward
                    var_nbrs = edge_rule == "AND" ? Set(neighbors(blacklist_graph, next_var)) : Set(neighbors(graph, next_var))
                else
                    var_nbrs = Set{Int}()
                end

                job = (next_var, all_univar_nbrs[next_var], HitonState{Int}('S', OrderedDict(), OrderedDict(), [], Dict()), var_nbrs)
                put!(shared_job_q, job)
                queued_jobs += 1

                if isempty(waiting_vars)
                    break
                end
            end
        end


        # print network stats after each update interval
        curr_time = time()
        if curr_time - last_update_time > update_interval
            if verbose
                println("\nTime passed: ", Int(round(curr_time - start_time)), ". Finished nodes: ", length(target_vars) - remaining_jobs, ". Remaining nodes: ", remaining_jobs)

                if check_convergence
                    println("Convergence times: $last_conv_time $(curr_time - last_conv_time - start_time) $((curr_time - last_conv_time - start_time) / last_conv_time) $(ne(graph) - last_conv_num_edges)")
                end

                print_network_stats(graph)
            end

            last_update_time = curr_time
        end

        if !isempty(output_folder) && curr_time - last_output_time > output_interval
            if temp_output_type == "single"
                curr_out_path = temp_out_path
            else
                curr_out_path = joinpath(output_folder, "tmp_network_" * string(now())[1:end-4] * ".edgelist")
            end

            verbose && println("Writing temporary graph to $curr_out_path")

            write_edgelist(curr_out_path, output_graph)
            last_output_time = curr_time
        end


        if convergence_threshold != 0.0 && !converged
            if !check_convergence && remaining_jobs / jobs_total <= conv_check_start
                check_convergence = true
                last_conv_time = curr_time - start_time
                last_conv_num_edges = ne(graph)

                verbose && println("Starting convergence checks at $last_conv_num_edges edges.")

            elseif check_convergence
                delta_time = (curr_time - start_time - last_conv_time) / last_conv_time

                if delta_time > conv_time_step
                    new_num_edges = ne(graph)
                    delta_num_edges = (new_num_edges - last_conv_num_edges) / last_conv_num_edges
                    conv_level = delta_num_edges / delta_time

                    verbose && println("Current convergence level: $conv_level")

                    if conv_level < convergence_threshold
                        converged = true
                        verbose && println("\tCONVERGED! Waiting for remaining processes to finish their current load.")
                    end

                    last_conv_time = curr_time - start_time
                    last_conv_num_edges = new_num_edges
                end
            end
        end


    end

    if !workers_local
        rmprocs(workers())
    end

    graph_dict
end

end
