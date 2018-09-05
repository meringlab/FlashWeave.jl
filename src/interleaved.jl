function interleaved_worker(data::AbstractMatrix{ElType}, levels, cor_mat, edge_rule::String,
     nonsparse_cond::Bool, shared_job_q::RemoteChannel, shared_result_q::RemoteChannel,
     GLL_args::Dict{Symbol,Any}) where {ElType<:Real}

    nonsparse_cond && @warn "nonsparse_cond currently not implemented"

    converged = false
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
                prev_state = HitonState('C', prev_state.state_results, prev_state.inter_results,
                                        prev_state.unchecked_vars, prev_state.state_rejections)
            end

            blacklist = Set{Int}()
            whitelist = skip_nbrs

            nbr_state = si_HITON_PC(target_var, data, levels, cor_mat; univar_nbrs=univar_nbrs,
                                    prev_state=prev_state, blacklist=blacklist, whitelist=whitelist, GLL_args...)

            put!(shared_result_q, (target_var, nbr_state))
        catch exc
            println("Exception occurred! ", exc)
            println(catch_stacktrace())
            throw(exc)
        end

    end
end


function interleaved_backend(target_vars::AbstractVector{Int}, data::AbstractMatrix{ElType},
        all_univar_nbrs::Dict{Int,OrderedDict{Int,Tuple{Float64,Float64}}}, levels::Vector{DiscType},
        cor_mat::Matrix{ContType}, GLL_args::Dict{Symbol,Any};
        update_interval::Real=30.0, variable_ids=nothing, meta_variable_mask=nothing,
        convergence_threshold::AbstractFloat=0.01,
        conv_check_start::AbstractFloat=0.1, conv_time_step::AbstractFloat=0.1, parallel::String="multi_il",
        edge_rule::String="OR", edge_merge_fun=maxweight, nonsparse_cond::Bool=false, verbose::Bool=true,
        workers_local::Bool=true, feed_forward::Bool=true) where {ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}

    test_name = GLL_args[:test_name]
    weight_type = GLL_args[:weight_type]
    jobs_total = length(target_vars)

    if startswith(parallel, "multi") || startswith(parallel, "threads")
        n_workers = nprocs() - 1
        job_q_buff_size = n_workers * 5
        worker_ids = workers()
        @assert n_workers > 0 "Need to add workers for parallel processing."
    elseif startswith(parallel, "single")
        n_workers = 1
        job_q_buff_size = 1
        worker_ids = [1]
    else
        error("$parallel not a valid execution mode.")
    end

    shared_job_q = RemoteChannel(() -> StackChannel{Tuple}(size(data, 2) * 2))
    shared_result_q = RemoteChannel(() -> Channel{Tuple}(size(data, 2)), 1)

    # initialize jobs
    queued_jobs = 0
    waiting_vars = Stack{Int}()
    for (i, target_var) in enumerate(reverse(target_vars))
        job = (target_var, all_univar_nbrs[target_var], HitonState{Int}('S', OrderedDict(), OrderedDict(),
                                                                        [], Dict()), Set{Int}())

        if i < jobs_total - job_q_buff_size
            push!(waiting_vars, target_var)
        else
            put!(shared_job_q, job)
            queued_jobs += 1
        end
    end

    verbose && println("\nPreparing workers..")

    worker_returns = [@spawnat wid interleaved_worker(data, levels, cor_mat, edge_rule,
                                                      nonsparse_cond,
                                                      shared_job_q, shared_result_q, GLL_args)
                      for wid in worker_ids]

    verbose && println("\nDone. Starting inference..")

    remaining_jobs = jobs_total
    n_vars = size(data, 2)
    graph_dict = Dict{Int, HitonState{Int}}()

    # this graph is just used for efficiently keeping track of graph stats during the run
    graph = Graph(n_vars)

    edge_set = Set{Tuple{Int,Int}}()
    kill_signals_sent = 0
    start_time = time()
    last_update_time = start_time
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
                    skip_nbrs = Set(neighbors(graph, target_var))
                else
                    skip_nbrs= Set{Int}()
                end

                job = (target_var, all_univar_nbrs[target_var], curr_state, skip_nbrs)
                put!(shared_job_q, job)
                queued_jobs += 1

            # node is complete
            else
                graph_dict[target_var] = curr_state

                for nbr in keys(curr_state.state_results)
                    add_edge!(graph, target_var, nbr)
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
                    var_nbrs = Set(neighbors(graph, next_var))
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

        if convergence_threshold != 0.0 && !converged
            if !check_convergence && remaining_jobs / jobs_total <= conv_check_start
                check_convergence = true
                global last_conv_time = curr_time - start_time
                global last_conv_num_edges = ne(graph)

                verbose && println("Starting convergence checks at $last_conv_num_edges edges.")

            elseif check_convergence
                delta_time = (curr_time - start_time - last_conv_time) / last_conv_time

                if delta_time > conv_time_step
                    new_num_edges = ne(graph)
                    delta_num_edges = (new_num_edges - last_conv_num_edges) / last_conv_num_edges
                    conv_level = delta_num_edges / delta_time

                    verbose && println("Latest convergence step change: $(round(conv_level, digits=5))")

                    if conv_level < convergence_threshold
                        converged = true
                        verbose && println("\tCONVERGED! Waiting for the remaining processes to finish their current load.")
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
