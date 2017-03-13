module Learning

export LGL, si_HITON_PC

using MultipleTesting
using DistributedArrays
using Cauocc.Tests
using Cauocc.Misc
using Cauocc.Statfuns
using Cauocc.StackChannels


function interleaving_phase(T::Int, candidates::Vector{Int}, data,
    test_name::String, max_k::Int, alpha::Float64, hps::Int=5, pwr::Float64=0.5, levels::Vector{Int}=Int[],
    data_row_inds::Vector{Int64}=Int64[], data_nzero_vals::Vector{Int64}=Int64[],
    prev_TPC_dict::Dict{Int,Tuple{Float64,Float64}}=Dict(), time_limit::Float64=0.0, start_time::Float64=0.0, debug::Int=0,
    whitelist::Set{Int}=Set{Int}())
        
    
    is_nz = is_zero_adjusted(test_name)
    
    if isempty(prev_TPC_dict)
        TPC = [candidates[1]]
        TPC_dict = Dict{Int,Tuple{Float64,Float64}}(TPC[1] => (-1.0, -1.0))
        OPEN = candidates[2:end]
    else
        TPC = collect(keys(prev_TPC_dict))
        TPC_dict = prev_TPC_dict
        OPEN = candidates
    end
    
    #start_time = time()
    n_candidates = length(OPEN)
    for (cand_index, candidate) in enumerate(OPEN)
        if debug > 0
            println("\tTesting candidate $candidate ($cand_index out of $n_candidates) conditioned on $TPC")
        end
            
        if !isempty(whitelist) && candidate in whitelist
            push!(TPC, candidate)
            TPC_dict[candidate] = (NaN64, NaN64)
            continue
        end
        
        if is_nz && !issparse(data)
            sub_data = @view data[data[:, candidate] .!= 0, :]
        else
            sub_data = data
        end

        test_result = test_subsets(T, candidate, TPC, sub_data, test_name, max_k, alpha, hps=hps, pwr=pwr, levels=levels, data_row_inds=data_row_inds, data_nzero_vals=data_nzero_vals)
        
        if issig(test_result, alpha)
            push!(TPC, candidate)
            TPC_dict[candidate] = (test_result.stat, test_result.pval)
            
            if debug > 0
                println("\taccepted: ", test_result)
            end
        elseif debug > 0
            println("\trejected: ", test_result)
        end
        
        if cand_index < n_candidates && stop_reached(start_time, time_limit)
            candidates_unchecked = OPEN[cand_index+1:end]
            return TPC_dict, candidates_unchecked
        end
    end
    
    TPC_dict, Int[]
end


function elimination_phase(T::Int, TPC::Vector{Int}, data, test_name::String,
    max_k::Int, alpha::Float64, fast_elim::Bool, hps::Int=5, pwr::Float64=0.5, levels::Vector{Int}=[],
    data_row_inds::Vector{Int64}=Int64[], data_nzero_vals::Vector{Int64}=Int64[],
    prev_PC_dict::Dict{Int,Tuple{Float64,Float64}}=Dict(), PC_unchecked::Vector{Int}=[],
    time_limit::Float64=0.0, start_time::Float64=0.0, debug::Int=0, whitelist::Set{Int}=Set{Int}())
    
    is_nz = is_zero_adjusted(test_name)
    
    if !isempty(PC_unchecked) && !isempty(prev_PC_dict)
        PC_dict = prev_PC_dict
        PC_candidates = PC_unchecked
    else
        PC_dict = Dict{Int,Tuple{Float64,Float64}}()
        PC_candidates = TPC
    end
    
    PC = copy(TPC)
    n_candidates = length(PC_candidates)
    
    if n_candidates < 1
        if n_candidates == 1
            PC_dict[PC_candidates[1]] = (NaN64, alpha - alpha * 0.1)
        end
        return PC_dict, Int[]
    end
        
    for (cand_index, candidate) in enumerate(PC_candidates)            
        PC_nocand = PC[PC .!= candidate]
        
        if debug > 0
            println("\tTesting candidate $candidate ($cand_index out of $n_candidates) conditioned on $PC_nocand")
        end
        
        if !isempty(whitelist) && candidate in whitelist
            PC_dict[candidate] = (NaN64, NaN64)
            continue
        end
            
        if is_nz && !issparse(data)
            sub_data = @view data[data[:, candidate] .!= 0, :]
        else
            sub_data = data
        end
        
        test_result = test_subsets(T, candidate, PC_nocand, sub_data, test_name, max_k, alpha, hps=hps, levels=levels, data_row_inds=data_row_inds, data_nzero_vals=data_nzero_vals)

        if !issig(test_result, alpha)
            deleteat!(PC, findin(PC, candidate))
            
            if debug > 0
                println("\trejected: ", test_result)
            end   
        else
            PC_dict[candidate] = (test_result.stat, test_result.pval)
            
            if debug > 0
                println("\taccepted: ", test_result)
            end
        end
        
        if cand_index < n_candidates && stop_reached(start_time, time_limit)
            TPC_unchecked = PC_candidates[cand_index+1:end]
            return PC_dict, TPC_unchecked
        end
    end

    PC_dict, Int[]
end


function si_HITON_PC(T, data; test_name::String="mi", max_k::Int=3, alpha::Float64=0.05, hps::Int=5,
    pwr::Float64=0.5, FDR::Bool=true, fast_elim::Bool=true, weight_type::String="cond_logpval", whitelist::Set{Int}=Set{Int}(),
    univar_nbrs::Dict{Int,Tuple{Float64,Float64}}=Dict{Int,Tuple{Float64,Float64}}(), levels::Vector{Int64}=Int64[],
    univar_step::Bool=true,
    prev_state::HitonState=HitonState("S", Dict(), []), debug::Int=0, time_limit::Float64=0.0)
    """
    # prepare input
    if typeof(target_var) == Int
        T = target_var
    elseif typeof(target_var) == String
        T = findfirst(header, target_var)   
    end
    """
    if debug > 0
        println("Finding neighbors for $T")
    end
    
    state = HitonState("S", Dict(), [])
    if is_zero_adjusted(test_name)
        if issparse(data)
            data = data[data[:, T] .!= 0, :]
        else
            data = @view data[data[:, T] .!= 0, :]
        end
    end
    
    if issparse(data) && isdiscrete(test_name)
        data_row_inds = rowvals(data)
        data_nzero_vals = nonzeros(data)
    else
        data_row_inds = Int64[]
        data_nzero_vals = Int64[]
    end
    
    test_variables = filter(x -> x != T, 1:size(data, 2))    
    start_time = time_limit > 0.0 ? time() : 0.0
    
    # univariate filtering
    if debug > 0
        println("UNIVARIATE")
    end
    
    if isdiscrete(test_name)
        if isempty(levels)
            levels = map(x -> get_levels(data[:, x]), 1:size(data, 2))
        end
        
        if levels[T] < 2
            return Dict{Int,Float64}()
        end
        
        if univar_step
            univar_test_results = test(T, test_variables, data, test_name, hps, levels, data_row_inds, data_nzero_vals)
        end
    else
        levels = Int[]
           
        if univar_step
            univar_test_results = test(T, test_variables, data, test_name)
        end
    end
    
    
    # if local FDR was specified, apply it here
    if univar_step
        pvals = map(x -> x.pval, univar_test_results)
    
        if FDR
            pvals = adjust(pvals, BenjaminiHochberg())
        end
    
        univar_nbrs = Dict([(nbr, (stat, pval)) for (nbr, stat, pval) in zip(test_variables, map(x -> x.stat, univar_test_results), pvals) if pval < alpha])
    end
        
    
    if debug > 0
        println("\t", collect(zip(test_variables, univar_nbrs)))
    end    
    
    # if conditioning should be performed
    if max_k > 0

        if prev_state.phase != "E"
        
            if prev_state.phase == "I"
                prev_TPC_dict = prev_state.state_results
                candidates = prev_state.unchecked_vars
            else
                # sort candidates
                candidate_pval_pairs = [(candidate, univar_nbrs[candidate][2]) for candidate in keys(univar_nbrs)]
                sort!(candidate_pval_pairs, by=x -> x[2])
                candidates = map(x -> x[1], candidate_pval_pairs)
                prev_TPC_dict = Dict{Int,Tuple{Float64,Float64}}()
            end

            if debug > 0
                println("\tnumber of candidates:", length(candidates), candidates[1:min(length(candidates), 20)])
                println("\nINTERLEAVING\n")
            end
                
            if isempty(candidates)
                return Dict{Int,Float64}()
            end
            
            # interleaving phase
            TPC_dict, candidates_unchecked = interleaving_phase(T, candidates, data, test_name, max_k, alpha, hps, pwr, levels, data_row_inds, data_nzero_vals, prev_TPC_dict, time_limit, start_time, debug, whitelist)
            
            # set test stats of the initial candidate to its univariate association results
            if prev_state.phase == "S"
                TPC_dict[candidates[1]] = univar_nbrs[candidates[1]]
            end
            
            if !isempty(candidates_unchecked)
                
                if debug > 0
                    println("Time limit exceeded, reporting incomplete results")
                end
                
                state.phase = "I"
                state.state_results = TPC_dict
                state.unchecked_vars = candidates_unchecked
                
                return state, univar_nbrs
            end

            if debug > 0
                println("After interleaving:", length(TPC_dict), " ", collect(keys(TPC_dict)))

                if debug > 1
                    println(TPC_dict)
                end

                println("\nELIMINATION\n")
            end
        end
        

        # elimination phase

        if prev_state.phase == "E"
            prev_PC_dict = prev_state.state_results
            PC_unchecked = prev_state.unchecked_vars
            PC_candidates = [keys(prev_PC_dict)..., PC_unchecked...]
        else
            prev_PC_dict = Dict{Int,Tuple{Float64,Float64}}()
            PC_unchecked = Int[]
            PC_candidates = collect(keys(TPC_dict))
        end
        PC_dict, TPC_unchecked = elimination_phase(T, PC_candidates, data, test_name, max_k, alpha, fast_elim, hps, pwr, levels, data_row_inds, data_nzero_vals, prev_PC_dict, PC_unchecked, time_limit, start_time, debug, whitelist)
            
        if !isempty(TPC_unchecked)
                
            if debug > 0
                println("Time limit exceeded, reporting incomplete results")
            end
                
            state.phase = "E"
            state.state_results = PC_dict
            state.unchecked_vars = TPC_unchecked
                
            return state, univar_nbrs
        end

        if debug > 1
            println(PC_dict)
        end
    else
        PC_dict = univar_nbrs
    end
    
    
    nbr_dict = make_weights(PC_dict, univar_nbrs, weight_type)
    
    nbr_dict
end


function LGL(data; test_name::String="mi", max_k::Int=3, alpha::Float64=0.05, hps::Int=5, pwr::Float64=0.5,
    FDR::Bool=true, global_univar::Bool=true, parallel::String="single", fast_elim::Bool=true, weight_type::String="cond_logpval", verbose::Bool=true,
    debug::Int=0, time_limit::Float64=0.0, header::Vector{String}=String[])
    """
    parallel: 'single', 'multi_e', 'multi_il'
    """
    if issparse(data)
        warn("Usage of sparse matrices still produces slightly inaccurate results. Use at own risk!")
    end
    
    kwargs = Dict(:test_name => test_name, :max_k => max_k, :alpha => alpha, :hps => hps, :pwr => pwr, :FDR => FDR,
    :fast_elim => fast_elim, :weight_type => weight_type, :univar_step => !global_univar, :debug => debug,
    :time_limit => time_limit)
    
    workers_local = workers_all_local()
    
    if time_limit != 0.0 && parallel != "multi_il"
        warn("Using time_limit without interleaved parallelism is not advised.")
    end
    
    if isdiscrete(test_name)
        if verbose
            println("Computing levels..")
        end
        
        levels = map(x -> get_levels(data[:, x]), 1:size(data, 2))
    else
        levels = Int64[]
    end
    
    if global_univar
        # precompute univariate associations and sort variables (fewest neighbors first)
        if verbose
            println("Computing univariate associations..")
        end
        
        all_univar_nbrs = pw_univar_neighbors(data; test_name=test_name, alpha=alpha, hps=hps, FDR=FDR, levels=levels, parallel=parallel, workers_local=workers_local)
        var_nbr_sizes = [(x, length(all_univar_nbrs[x])) for x in 1:size(data, 2)]
        target_vars = [nbr_size_pair[1] for nbr_size_pair in sort(var_nbr_sizes, by=x -> x[2])]
    else
        target_vars = 1:size(data, 2)
        all_univar_nbrs = Dict([(x, Dict{Int,Tuple{Float64,Float64}}()) for x in target_vars])
    end
    
    
    if verbose
        println("Running si_HITON_PC for each variable..")
    end
    
    if parallel == "single" || nprocs() == 1
        nbr_dicts = [si_HITON_PC(x, data; univar_nbrs=all_univar_nbrs[x], levels=levels, kwargs...) for x in target_vars]
        graph_dict = Dict([(target_var, nbr_dict) for (target_var, nbr_dict) in zip(target_vars, nbr_dicts)])
    else
        # embarassingly parallel
        if parallel == "multi_ep"
            @sync nbr_dict_refs = [@spawn si_HITON_PC(x, data; univar_nbrs=all_univar_nbrs[x], levels=levels, kwargs...) for x in target_vars]

            nbr_dicts = map(fetch, nbr_dict_refs)
            graph_dict = Dict([(target_var, nbr_dict) for (target_var, nbr_dict) in zip(target_vars, nbr_dicts)])
            
        # interleaved parallelism
        elseif parallel == "multi_il"
            graph_dict = interleaved_backend(target_vars, data, all_univar_nbrs, levels, kwargs)
        else
            error("'$parallel' not a valid parallel mode")
        end
        
    end
    
            
    if verbose
        println("Postprocessing..")
    end
    
    if !isempty(header)
        graph_dict = Dict([(header[x], Dict([(header[y], graph_dict[x][y]) for y in keys(graph_dict[x])])) for x in keys(graph_dict)])
    end
    
    convert(Dict{Union{Int, String},Dict{Union{Int, String}, Float64}}, graph_dict)
end


# SPECIALIZED FUNCTIONS AND TYPES

function pw_univar_kernel!(X, data, stats, pvals, test_name, hps, levels, nz, data_row_inds, data_nzero_vals)
    adj_factor = nz ? 1 : 0
    
    n_vars = size(data, 2)
    
    if nz && !issparse(data)
        sub_data = @view data[data[:, X] .!= 0, :]
    else
        sub_data = data
    end
        
    Ys = collect(X+1:n_vars)
    
    if isdiscrete(test_name)
        test_results = test(X, Ys, sub_data, test_name, hps, levels, data_row_inds, data_nzero_vals)
    else
        test_results = test(X, Ys, sub_data, test_name)
    end
    
    for (Y, test_res) in zip(Ys, test_results)
        pair_index = sum(n_vars-1:-1:n_vars-X) - n_vars + Y
        stats[pair_index] = test_res.stat
        pvals[pair_index] = test_res.pval
    end   
end


function pw_univar_kernel(X, data, test_name, hps, levels, nz, data_row_inds, data_nzero_vals)
    adj_factor = nz ? 1 : 0
    
    n_vars = size(data, 2)
    
    if nz && !issparse(data)
        sub_data = @view data[data[:, X] .!= 0, :]
    else
        sub_data = data
    end
        
    Ys = collect(X+1:n_vars)
    test_results = test(X, Ys, sub_data, test_name, hps, levels, data_row_inds, data_nzero_vals) 
end


function pw_univar_neighbors(data; test_name::String="mi", alpha::Float64=0.05, hps::Int=5, FDR::Bool=true, levels::Vector{Int64}=Int64[], parallel="single", workers_local::Bool=true)
    
    n_vars = size(data, 2)
    n_pairs = convert(Int, n_vars * (n_vars - 1) / 2)
    #levels = map(x -> length(unique(data[:, x])), 1:size(data, 2))
    
    nz = is_zero_adjusted(test_name)
    
    if issparse(data) && isdiscrete(test_name)
        data_row_inds = rowvals(data)
        data_nzero_vals = nonzeros(data)
    else
        data_row_inds = Int64[]
        data_nzero_vals = Int64[]
    end
    
    if parallel == "single"
        pvals = zeros(Float64, n_pairs)
        stats = zeros(Float64, n_pairs)
        
        for X in 1:n_vars-1
            pw_univar_kernel!(X, data, stats, pvals, test_name, hps, levels, nz, data_row_inds, data_nzero_vals)
        end
        
    elseif startswith(parallel, "multi")
        # if worker processes are on the same machine, use local memory sharing via shared arrays
        if workers_local   
            shared_pvals = SharedArray(Float64, n_pairs)
            shared_stats = SharedArray(Float64, n_pairs)
            @sync @parallel for X in 1:n_vars-1
                pw_univar_kernel!(X, data, shared_stats, shared_pvals, test_name, hps, levels, nz, data_row_inds, data_nzero_vals)
            end
            stats = shared_stats.s
            pvals = shared_pvals.s
        
        # otherwise make workers store test results remotely and gather them in the end via network
        else
            @sync all_test_result_refs = [@spawn pw_univar_kernel(X, data, test_name, hps, levels, nz, data_row_inds, data_nzero_vals) for X in 1:n_vars-1]
            all_test_results = map(fetch, all_test_result_refs)
            pvals = zeros(Float64, n_pairs)
            stats = zeros(Float64, n_pairs)
          
            i = 1
            for test_res in all_test_results
                for t in test_res
                    stats[i] = t.stat
                    pvals[i] = t.pval
                    i += 1
                end
            end
        end
        
    elseif startswith(parallel, "threads")
        pvals = zeros(Float64, n_pairs)
        stats = zeros(Float64, n_pairs)
        Threads.@threads for X in 1:n_vars-1
            for X in 1:n_vars-1
                pw_univar_kernel!(X, data, stats, pvals, test_name, hps, levels, nz)
            end
        end
    else
        error("'$parallel' is not a valid parallel mode")
    end
    
    if FDR
        pvals = adjust(pvals, BenjaminiHochberg())
    end
    
    nbr_dict = Dict([(X, Dict{Int,Tuple{Float64,Float64}}()) for X in 1:n_vars])
    for X in 1:n_vars-1, Y in X+1:n_vars
        pair_index = sum(n_vars-1:-1:n_vars-X) - n_vars + Y
        pval = pvals[pair_index]
        
        if pval < alpha
            stat = stats[pair_index]
            nbr_dict[X][Y] = (stat, pval)
            nbr_dict[Y][X] = (stat, pval)
        end
    end
    
    
    
    nbr_dict
end


function interleaved_worker(data, levels, shared_job_q::RemoteChannel, shared_result_q::RemoteChannel, GLL_fun, GLL_args::Dict{Symbol,Any})
    while true
        try
            target_var, univar_nbrs, prev_state, current_nbrs = take!(shared_job_q)
            
            # if kill signal
            if target_var == -1
                return
            end

            nbr_result = si_HITON_PC(target_var, data; univar_nbrs=univar_nbrs, levels=levels, prev_state=prev_state, whitelist=current_nbrs, GLL_args...)
            put!(shared_result_q, (target_var, nbr_result))
        catch exc
            put!(shared_result_q, (target_var, exc))
        end
        
    end
end


function interleaved_backend(target_vars::Vector{Int}, data, all_univar_nbrs, levels, GLL_args)
    n_workers = nprocs() - 1
    shared_job_q = RemoteChannel(() -> StackChannel{Tuple}(size(data, 2) * 2), 1)
    shared_result_q = RemoteChannel(() -> Channel{Tuple}(size(data, 2)), 1)
    
    # initialize jobs
    for target_var in reverse(target_vars)
        job = (target_var, all_univar_nbrs[target_var], HitonState("S", Dict(), []), Set{Int}())
        put!(shared_job_q, job)
    end
    
    worker_returns = [@spawn interleaved_worker(data, levels, shared_job_q, shared_result_q, si_HITON_PC, GLL_args) for x in 1:n_workers]
    
    remaining_jobs = length(target_vars)
    graph_dict = Dict{Int,Dict{Int,Float64}}([(target_var, Dict{Int,Float64}()) for target_var in target_vars])
    kill_signals_sent = 0
    while remaining_jobs > 0
        target_var, nbr_result = take!(shared_result_q)
            
        if typeof(nbr_result) <: Tuple
            current_nbrs = graph_dict[target_var]
            job = (target_var, nbr_result[2], nbr_result[1], Set(keys(current_nbrs)))
            put!(shared_job_q, job)
        elseif typeof(nbr_result) <: Dict
            nbr_dict = nbr_result
            for nbr in keys(nbr_dict)
                graph_dict[target_var][nbr] = nbr_dict[nbr]
            end
            remaining_jobs -= 1
                
            # kill workers if not needed anymore
            if remaining_jobs < n_workers
                kill_signal = (-1, Dict{Int,Tuple{Float64,Float64}}(), HitonState("S", Dict(), []), Set{Int}())
                put!(shared_job_q, kill_signal)
                kill_signals_sent += 1
            end
        else
            throw(nbr_result)
        end
    end
    
    graph_dict
end


end
