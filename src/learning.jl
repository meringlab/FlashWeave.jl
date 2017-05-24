module Learning

export LGL, si_HITON_PC

using MultipleTesting
using LightGraphs
using DataStructures
using StatsBase

using Cauocc.Tests
using Cauocc.Misc
using Cauocc.Statfuns
using Cauocc.StackChannels


function interleaving_phase(T::Int, candidates::Vector{Int}, data,
    test_name::String, max_k::Int, alpha::Float64, hps::Int=5, pwr::Float64=0.5, levels::Vector{Int}=Int[],
    data_row_inds::Vector{Int64}=Int64[], data_nzero_vals::Vector{Int64}=Int64[],
    prev_TPC_dict::Dict{Int,Tuple{Float64,Float64}}=Dict(), time_limit::Float64=0.0, start_time::Float64=0.0, debug::Int=0,
    whitelist::Set{Int}=Set{Int}(), blacklist::Set{Int}=Set{Int}(), cor_mat::Matrix{Float64}=zeros(Float64, 0, 0),
    pcor_set_dict::Dict{String,Dict{String,Float64}}=Dict{String,Dict{String,Float64}}())
        
    
    is_nz = is_zero_adjusted(test_name)
    is_discrete = isdiscrete(test_name)
    is_dense = !issparse(data)
    
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
        
        if !isempty(blacklist) && candidate in blacklist
            continue
        end
        
        if is_nz && is_dense && (!is_discrete || levels[candidate] > 2)
            sub_data = @view data[data[:, candidate] .!= 0, :]
        else
            sub_data = data
        end

        test_result = test_subsets(T, candidate, TPC, sub_data, test_name, max_k, alpha, hps=hps, pwr=pwr, levels=levels, data_row_inds=data_row_inds, data_nzero_vals=data_nzero_vals, cor_mat=cor_mat, pcor_set_dict=pcor_set_dict)
        
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
    max_k::Int, alpha::Float64, hps::Int=5, pwr::Float64=0.5, levels::Vector{Int}=[],
    data_row_inds::Vector{Int64}=Int64[], data_nzero_vals::Vector{Int64}=Int64[],
    prev_PC_dict::Dict{Int,Tuple{Float64,Float64}}=Dict(), PC_unchecked::Vector{Int}=[],
    time_limit::Float64=0.0, start_time::Float64=0.0, debug::Int=0, whitelist::Set{Int}=Set{Int}(),
    blacklist::Set{Int}=Set{Int}(), cor_mat::Matrix{Float64}=zeros(Float64, 0, 0),
    pcor_set_dict::Dict{String,Dict{String,Float64}}=Dict{String,Dict{String,Float64}}())
    
    is_nz = is_zero_adjusted(test_name)
    is_discrete = isdiscrete(test_name)
    is_dense = !issparse(data)    
            
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
        
        if !isempty(blacklist) && candidate in blacklist
            continue
        end
            
        if is_nz && is_dense && (!is_discrete || levels[candidate] > 2)
            sub_data = @view data[data[:, candidate] .!= 0, :]
        else
            sub_data = data
        end
        
        test_result = test_subsets(T, candidate, PC_nocand, sub_data, test_name, max_k, alpha, hps=hps, levels=levels, data_row_inds=data_row_inds, data_nzero_vals=data_nzero_vals, cor_mat=cor_mat, pcor_set_dict=pcor_set_dict)

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


function si_HITON_PC(T, data; test_name::String="mi", max_k::Int=3, alpha::Float64=0.01, hps::Int=5,
    pwr::Float64=0.5, FDR::Bool=true, weight_type::String="cond_logpval", whitelist::Set{Int}=Set{Int}(),
        blacklist::Set{Int}=Set{Int}(),
        univar_nbrs::Dict{Int,Tuple{Float64,Float64}}=Dict{Int,Tuple{Float64,Float64}}(), levels::Vector{Int64}=Int64[],
    univar_step::Bool=true, cor_mat::Matrix{Float64}=zeros(Float64, 0, 0),
    pcor_set_dict::Dict{String,Dict{String,Float64}}=Dict{String,Dict{String,Float64}}(),
    prev_state::HitonState=HitonState("S", Dict(), []), debug::Int=0, time_limit::Float64=0.0)

    if debug > 0
        println("Finding neighbors for $T")
    end
    
    state = HitonState("S", Dict(), [])
    
    if isdiscrete(test_name)
        if isempty(levels)
            levels = map(x -> get_levels(data[:, x]), 1:size(data, 2))
        end
        
        if levels[T] < 2
            state.phase = "F"
            state.state_results = Dict{Int,Tuple{Float64,Float64}}()
            state.unchecked_vars = Int64[]
            return state
        end    
    else
        levels = Int[]
    end
        

    if is_zero_adjusted(test_name)    
        if !isdiscrete(test_name) || levels[T] > 2
            if issparse(data)
                data = data[data[:, T] .!= 0, :]
                #levels = map(x -> get_levels(data[:, x]), 1:size(data, 2))
            else
                data = @view data[data[:, T] .!= 0, :]
                #levels = map(x -> get_levels(data[:, x]), 1:size(data, 2))
            end
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
    
    if univar_step
        if isdiscrete(test_name)
            univar_test_results = test(T, test_variables, data, test_name, hps, levels, data_row_inds, data_nzero_vals)
        else
            univar_test_results = test(T, test_variables, data, test_name, cor_mat)
        end
    end
    
    
    # if local FDR was specified, apply it here
    if univar_step
        pvals = map(x -> x.pval, univar_test_results)
    
        if FDR
            #pvals = adjust(pvals, BenjaminiHochberg())
            pvals = benjamini_hochberg(pvals)
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
                state.phase = "F"
                state.state_results = Dict{Int,Tuple{Float64,Float64}}()
                state.unchecked_vars = Int64[]
                return state#Dict{Int,Float64}()
            end
            
            # interleaving phase
            TPC_dict, candidates_unchecked = interleaving_phase(T, candidates, data, test_name, max_k, alpha, hps, pwr, levels, data_row_inds, data_nzero_vals, prev_TPC_dict, time_limit, start_time, debug, whitelist, blacklist, cor_mat, pcor_set_dict)
            
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
                
                return state
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
        PC_dict, TPC_unchecked = elimination_phase(T, PC_candidates, data, test_name, max_k, alpha, hps, pwr, levels, data_row_inds, data_nzero_vals, prev_PC_dict, PC_unchecked, time_limit, start_time, debug, whitelist, blacklist, cor_mat, pcor_set_dict)
            
        if !isempty(TPC_unchecked)
                
            if debug > 0
                println("Time limit exceeded, reporting incomplete results")
            end
                
            state.phase = "E"
            state.state_results = PC_dict
            state.unchecked_vars = TPC_unchecked
                
            return state
        end

        if debug > 1
            println(PC_dict)
        end
    else
        PC_dict = univar_nbrs
    end
    
    
    #nbr_dict = make_weights(PC_dict, univar_nbrs, weight_type)
    #
    #nbr_dict
    state.phase = "F"
    state.state_results = PC_dict
    state.unchecked_vars = Int64[]
    
    state
end


function LGL(data; test_name::String="mi", max_k::Int=3, alpha::Float64=0.01, hps::Int=5, pwr::Float64=0.5,
    convergence_threshold::Float64=0.01, FDR::Bool=true, global_univar::Bool=true, parallel::String="single",
        fast_elim::Bool=true, precluster_sim::Float64=0.0,
        weight_type::String="cond_logpval", edge_rule::String="OR", verbose::Bool=true, update_interval::Float64=30.0,
        edge_merge_fun=maxweight,
    debug::Int=0, time_limit::Float64=-1.0, header::Vector{String}=String[], recursive_pcor::Bool=true)
    """
    time_limit: -1.0 set heuristically, 0.0 no time_limit, otherwise time limit in seconds
    parallel: 'single', 'single_il', 'multi_ep', 'multi_il'
    fast_elim: currently always on
    """
    
    kwargs = Dict(:test_name => test_name, :max_k => max_k, :alpha => alpha, :hps => hps, :pwr => pwr, :FDR => FDR,
    :weight_type => weight_type, :univar_step => !global_univar, :debug => debug,
    :time_limit => time_limit)
    
    if time_limit == -1.0
        if parallel == "multi_il"
            time_limit = log2(size(data, 2))
            println("Setting 'time_limit' to $time_limit s.")
        else
            time_limit = 0.0
        end
    end
        
    workers_local = workers_all_local()
    
    if time_limit != 0.0 && parallel != "multi_il"
        warn("Using time_limit without interleaved parallelism is not advised.")
    elseif parallel == "multi_il" && time_limit == 0.0
        warn("Specify 'time_limit' when using interleaved parallelism to increase speed.")
    end
    
    if recursive_pcor && iscontinuous(test_name)
        warn("setting 'recursive_pcor' to true produces different results in case of perfectly correlated variables, caution advised")
        if max_k == 0
            warn("Set 'recursive_pcor' to false when only computing univariate associations ('max_k' == 0) to gain speed.")
        end
    end
    
    if isdiscrete(test_name)
        if verbose
            println("Computing levels..")
        end
        levels = map(x -> get_levels(data[:, x]), 1:size(data, 2))
        cor_mat = zeros(Float64, 0, 0)
    else
        levels = Int64[]
        if recursive_pcor
            cor_mat = is_zero_adjusted(test_name) ? cor_nz(data) : cor(data)
        else
            cor_mat = zeros(Float64, 0, 0)
        end
    end
    
    pcor_set_dict = Dict{String,Dict{String,Float64}}()
    if global_univar
        # precompute univariate associations and sort variables (fewest neighbors first)
        if verbose
            println("Computing univariate associations..")
        end
        
        all_univar_nbrs = pw_univar_neighbors(data; test_name=test_name, alpha=alpha, hps=hps, FDR=FDR, levels=levels, parallel=parallel, workers_local=workers_local, cor_mat=cor_mat)
        var_nbr_sizes = [(x, length(all_univar_nbrs[x])) for x in 1:size(data, 2)]
        target_vars = [nbr_size_pair[1] for nbr_size_pair in sort(var_nbr_sizes, by=x -> x[2])]
        
        if verbose
            println("\nUnivariate degree stats:")
            nbr_nums = map(length, values(all_univar_nbrs))
            println(summarystats(nbr_nums), "\n")
            if mean(nbr_nums) > size(data, 1) * 0.2
                warn("The univariate network is exceptionally dense, computations may be very slow. Check if appropriate normalization was used (employ niche-mode if not yet the case) and try using the AND rule to gain speed.")
            end
        end
    else
        target_vars = 1:size(data, 2)
        all_univar_nbrs = Dict([(x, Dict{Int,Tuple{Float64,Float64}}()) for x in target_vars])
    end
    
    
    if precluster_sim != 0.0
        if verbose
            println("Clustering..")
        end
        
        univar_matrix = pw_unistat_matrix(data, test_name; pw_stat_dict=all_univar_nbrs)
        clust_repres, clust_dict = cluster_data(data, test_name; cluster_sim_threshold=precluster_sim, sim_mat=univar_matrix)
        
        if verbose
            println("\tfound $(length(clust_repres)) clusters")
        end
        
        target_vars = clust_repres
        data = data[:, clust_repres]
        
        target_vars = collect(1:length(clust_repres))
        var_clust_dict = Dict(zip(clust_repres, 1:length(clust_repres)))
        clust_var_dict = Dict(zip(1:length(clust_repres), clust_repres))
        all_univar_nbrs = map_edge_keys(all_univar_nbrs, var_clust_dict)
        
        if isdiscrete(test_name)
            levels = levels[clust_repres]
        end
    end
    
    
    if verbose
        println("Running si_HITON_PC for each variable..")
    end
        
    if max_k == 0 && global_univar
        nbr_dict = all_univar_nbrs
    else
        if parallel == "single" || nprocs() == 1
            nbr_results = [si_HITON_PC(x, data; univar_nbrs=all_univar_nbrs[x], levels=levels, cor_mat=cor_mat, pcor_set_dict=pcor_set_dict, kwargs...) for x in target_vars]
        else
            # embarassingly parallel
            if parallel == "multi_ep"
                @sync nbr_results_refs = [@spawn si_HITON_PC(x, data; univar_nbrs=all_univar_nbrs[x], levels=levels, cor_mat=cor_mat, pcor_set_dict=pcor_set_dict, kwargs...) for x in target_vars]

                nbr_results = map(fetch, nbr_res_refs)

            # interleaved parallelism
            elseif endswith(parallel, "il")
                il_dict = interleaved_backend(target_vars, data, all_univar_nbrs, levels, update_interval, kwargs,
                                               convergence_threshold, cor_mat, parallel=parallel, edge_rule=edge_rule)
                nbr_results = [il_dict[target_var] for target_var in target_vars]
            else
                error("'$parallel' not a valid parallel mode")
            end
        
        end

        #if !endswith(parallel, "il")
        nbr_dict = Dict([(target_var, nbr_state.state_results) for (target_var, nbr_state) in zip(target_vars, nbr_results)])
    end
            
    if verbose
        println("Postprocessing..")
    end
    
    if precluster_sim != 0.0
        nbr_dict = map_edge_keys(nbr_dict, clust_var_dict)
        all_univar_nbrs = map_edge_keys(all_univar_nbrs, clust_var_dict)
    end
     
    weights_dict = Dict([(target_var, make_weights(nbr_dict[target_var], all_univar_nbrs[target_var], weight_type)) for target_var in keys(nbr_dict)])

    graph_dict = make_graph_symmetric(weights_dict, edge_rule)
    
    if precluster_sim != 0.0
        for (clust_repres, clust_members) in clust_dict
            for member in clust_members                
                graph_dict[member] = Dict{Int64,Float64}()
                
                for nbr in keys(graph_dict[clust_repres])
                    graph_dict[member][nbr] = graph_dict[clust_repres][nbr]
                end
                
                graph_dict[member][clust_repres] = NAN64
                graph_dict[clust_repres][member] = NAN64
            end
        end
    end

    #if !isempty(header)
    #    graph_dict = Dict([(header[x], Dict([(header[y], graph_dict[x][y]) for y in keys(graph_dict[x])])) for x in keys(graph_dict)])
    #end
    #convert(Dict{Union{Int, String},Dict{Union{Int, String}, Float64}}, graph_dict)
end


# SPECIALIZED FUNCTIONS AND TYPES

function pw_univar_kernel!(X, data, stats, pvals, test_name, hps, levels, nz, data_row_inds, data_nzero_vals, cor_mat)    
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
        test_results = test(X, Ys, sub_data, test_name, cor_mat)
    end
    
    for (Y, test_res) in zip(Ys, test_results)
        pair_index = sum(n_vars-1:-1:n_vars-X) - n_vars + Y
        stats[pair_index] = test_res.stat
        pvals[pair_index] = test_res.pval
    end   
end


function pw_univar_kernel(X, data, test_name, hps, levels, nz, data_row_inds, data_nzero_vals, cor_mat)    
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
        test_results = test(X, Ys, sub_data, test_name, cor_mat)
    end
end


function condensed_stats_to_dict(n_vars::Int64, pvals::Vector{Float64}, stats::Vector{Float64}, alpha::Float64)
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


function pw_univar_neighbors(data; test_name::String="mi", alpha::Float64=0.05, hps::Int=5, FDR::Bool=true,
        levels::Vector{Int64}=Int64[], parallel::String="single", workers_local::Bool=true,
        cor_mat::Matrix{Float64}=zeros(Float64, 0, 0))
    
    if startswith(test_name, "mi") && isempty(levels)
        levels = map(x -> get_levels(data[:, x]), 1:size(data, 2))
    end
    
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
    
    if startswith(parallel, "single")
        pvals = zeros(Float64, n_pairs)
        stats = zeros(Float64, n_pairs)
        
        for X in 1:n_vars-1
            pw_univar_kernel!(X, data, stats, pvals, test_name, hps, levels, nz, data_row_inds, data_nzero_vals, cor_mat)
        end
        
    elseif startswith(parallel, "multi")
        # if worker processes are on the same machine, use local memory sharing via shared arrays
        if workers_local   
            shared_pvals = SharedArray(Float64, n_pairs)
            shared_stats = SharedArray(Float64, n_pairs)
            @sync @parallel for X in 1:n_vars-1
                pw_univar_kernel!(X, data, shared_stats, shared_pvals, test_name, hps, levels, nz, data_row_inds,
                    data_nzero_vals, cor_mat)
            end
            stats = shared_stats.s
            pvals = shared_pvals.s
        
        # otherwise make workers store test results remotely and gather them in the end via network
        else
            @sync all_test_result_refs = [@spawn pw_univar_kernel(X, data, test_name, hps, levels, nz, data_row_inds, data_nzero_vals, cor_mat) for X in 1:n_vars-1]
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
                pw_univar_kernel!(X, data, stats, pvals, test_name, hps, levels, nz, data_row_inds, data_nzero_vals)
            end
        end
    #else
    #    error("'$parallel' is not a valid parallel mode")
    end
    
    if FDR
        #println("num pvals:", length(pvals))
        pvals = benjamini_hochberg(pvals)
        #pvals = adjust(pvals, BenjaminiHochberg())
    end
    
    condensed_stats_to_dict(n_vars, pvals, stats, alpha)
end


function pw_unistat_matrix(data, test_name::String; parallel::String="single",
        pw_stat_dict::Dict{Int64,Dict{Int64,Tuple{Float64,Float64}}}=Dict{Int64,Dict{Int64,Tuple{Float64,Float64}}}())
    #workers_local = nprocs() > 1 ? workers_all_local() : true
    #test_name = nz ? "mi_nz" : "mi"
    if isempty(pw_stat_dict)
        pw_stat_dict = pw_univar_neighbors(data, test_name=test_name, parallel=parallel)
    end
    
    stat_mat = zeros(Float64, size(data, 2), size(data, 2))
    
    for var_A in keys(pw_stat_dict)
        for var_B in keys(pw_stat_dict[var_A])
            (stat, pval) = pw_stat_dict[var_A][var_B]
            stat_mat[var_A, var_B] = stat
        end
    end
    stat_mat
end


function cluster_data(data, stat_type::String="fz"; cluster_sim_threshold::Float64=0.8, parallel="single",
    ordering="size", sim_mat::Matrix{Float64}=zeros(Float64, 0, 0))
    
    if isempty(sim_mat)
        sim_mat = pw_unistat_matrix(data, stat_type, parallel=parallel)
    end
    
    if stat_type == "mi"
        # can potentially be improved by pre-allocation
        entrs = mapslices(x -> entropy(counts(x) ./ length(x)), data, 1)
    elseif stat_type == "mi_nz"
        nz_mask = data .!= 0
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
            for var_B in unclustered_vars
                sim = sim_mat[var_A, var_B]
                
                if sim == 0.0
                    continue
                end
                
                if startswith(stat_type, "mi")
                    if stat_type == "mi"
                        entr_A = entrs[var_A]
                        entr_B = entrs[var_B]
                    elseif stat_type == "mi_nz"
                        curr_nz_mask = (nz_mask[:, var_A] & nz_mask[:, var_B])[:]
                        nz_elems = sum(curr_nz_mask)
                        entr_A = entropy(counts(data[curr_nz_mask, var_A]) ./ nz_elems)
                        entr_B = entropy(counts(data[curr_nz_mask, var_B]) ./ nz_elems)
                    end
                    norm_term = sqrt(entr_A * entr_B)
                    
                    sim = norm_term != 0.0 ? abs(sim) / norm_term : 0.0
                end
                
                if sim > cluster_sim_threshold
                    push!(clust_members, var_B)
                    pop!(unclustered_vars, var_B)
                end
            end
            clust_dict[var_A] = clust_members
        end
    end
    
    (sort(collect(keys(clust_dict))), clust_dict)                 
end


function interleaved_worker(data, levels, cor_mat, edge_rule, shared_job_q::RemoteChannel, shared_result_q::RemoteChannel, GLL_fun, GLL_args::Dict{Symbol,Any})
    while true
        try
            target_var, univar_nbrs, prev_state, skip_nbrs = take!(shared_job_q)
            # if kill signal
            if target_var == -1
                return
            end

            if edge_rule == "AND"
                nbr_state = si_HITON_PC(target_var, data; univar_nbrs=univar_nbrs, levels=levels, prev_state=prev_state, blacklist=skip_nbrs, cor_mat=cor_mat, GLL_args...)
            else
                nbr_state = si_HITON_PC(target_var, data; univar_nbrs=univar_nbrs, levels=levels, prev_state=prev_state, whitelist=skip_nbrs, cor_mat=cor_mat, GLL_args...)
            end
            put!(shared_result_q, (target_var, nbr_state))
        catch exc
            println("Exception occurred! ", exc)
            put!(shared_result_q, (target_var, exc))
        end
        
    end
end

function interleaved_backend(target_vars::Vector{Int}, data, all_univar_nbrs, levels, update_interval, GLL_args,
        convergence_threshold, cor_mat; conv_check_start=0.1, conv_time_step=0.1, parallel="multi", edge_rule="OR")
    jobs_total = length(target_vars)
    
    if startswith(parallel, "multi")
        n_workers = nprocs() - 1
        job_q_buff_size = n_workers * 2
        @assert n_workers > 0 "Need to add workers for parallel processing."
    elseif startswith(parallel, "single")
        n_workers = 1
        job_q_buff_size = 1
        @assert nprocs() > 1 "Need to have one additional worker for interleaved mode."
    else
        error("$parallel not a valid execution mode.")
    end
    
    shared_job_q = RemoteChannel(() -> StackChannel{Tuple}(size(data, 2) * 2), 1)
    shared_result_q = RemoteChannel(() -> Channel{Tuple}(size(data, 2)), 1)
    
    
    # initialize jobs
    queued_jobs = 0
    waiting_vars = Stack(Int)
    for (i, target_var) in enumerate(reverse(target_vars))
        job = (target_var, all_univar_nbrs[target_var], HitonState("S", Dict(), []), Set{Int}())
        
        if i < jobs_total - n_workers
            push!(waiting_vars, target_var)
        else
            put!(shared_job_q, job)
            queued_jobs += 1
        end
    end
            
    
    worker_returns = [@spawn interleaved_worker(data, levels, cor_mat, edge_rule, shared_job_q, shared_result_q, si_HITON_PC, GLL_args) for x in 1:n_workers]
    
    remaining_jobs = jobs_total
    
    graph_dict = Dict{Int64, HitonState}()
    
    # this graph is just used for efficiently keeping track of graph stats during the run
    graph = Graph(length(target_vars))
    
    if edge_rule == "AND"
        blacklist_graph = Graph(length(target_vars))
    end
    
    edge_set = Set{Tuple{Int64,Int64}}()
    kill_signals_sent = 0
    start_time = time()
    last_update_time = start_time
    check_convergence = false
    converged = false
    
    
    while remaining_jobs > 0
        target_var, nbr_result = take!(shared_result_q)
        queued_jobs -= 1
        if isa(nbr_result, HitonState)
            curr_state = nbr_result
            
            # node has not yet finished computing
            if curr_state.phase != "F"
                if converged
                    curr_state.unchecked_vars = Int64[]
                end

                skip_nbrs = edge_rule == "AND" ? Set(neighbors(blacklist_graph, target_var)) : Set(neighbors(graph, target_var))
                job = (target_var, all_univar_nbrs[target_var], curr_state, skip_nbrs)
                put!(shared_job_q, job)
                queued_jobs += 1
                
            # node is complete
            else
                graph_dict[target_var] = curr_state
                
                for nbr in keys(curr_state.state_results)
                    add_edge!(graph, target_var, nbr)
                end
                
                if edge_rule == "AND"
                    for a_var in target_vars
                        if !haskey(curr_state.state_results, a_var)
                            add_edge!(blacklist_graph, target_var, a_var)
                        end
                    end
                end
                
                remaining_jobs -= 1

                # kill workers if not needed anymore
                if remaining_jobs < n_workers
                    kill_signal = (-1, Dict{Int,Tuple{Float64,Float64}}(), HitonState("S", Dict(), []), Set{Int}())
                    put!(shared_job_q, kill_signal)
                    kill_signals_sent += 1
                end
            end
        else
            println(nbr_result)
            throw(nbr_result)
        end
        
        if !isempty(waiting_vars) && queued_jobs < job_q_buff_size
            for i in 1:job_q_buff_size - queued_jobs
                next_var = pop!(waiting_vars)
                var_nbrs = edge_rule == "AND" ? Set(neighbors(blacklist_graph, next_var)) : Set(neighbors(graph, next_var))
                
                job = (next_var, all_univar_nbrs[next_var], HitonState("S", Dict(), []), var_nbrs)
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
            println("\nTime passed: ", curr_time - start_time, ". Finished nodes: ", length(target_vars) - remaining_jobs, ". Remaining nodes: ", remaining_jobs)
            if check_convergence
                println("Convergence times: $last_conv_time $(curr_time - last_conv_time - start_time) $((curr_time - last_conv_time - start_time) / last_conv_time) $(ne(graph) - last_conv_num_edges)")
            end
            print_network_stats(graph)
            last_update_time = curr_time
        end
        
        
        if convergence_threshold != 0.0 && !converged
            if !check_convergence && remaining_jobs / jobs_total <= conv_check_start
                check_convergence = true
                last_conv_time = curr_time - start_time
                last_conv_num_edges = ne(graph)
                println("Starting convergence checks at $last_conv_num_edges edges.")
            elseif check_convergence
                delta_time = (curr_time - start_time - last_conv_time) / last_conv_time
                
                if delta_time > conv_time_step
                    new_num_edges = ne(graph)
                    delta_num_edges = (new_num_edges - last_conv_num_edges) / last_conv_num_edges
                    conv_level = delta_num_edges / delta_time
                    println("Current convergence level: $conv_level")
                    
                    if conv_level < convergence_threshold
                        converged = true
                        println("\tCONVERGED!")
                    end
                    
                    last_conv_time = curr_time - start_time
                    last_conv_num_edges = new_num_edges               
                end
            end
        end
        

    end
    
    graph_dict
end


end