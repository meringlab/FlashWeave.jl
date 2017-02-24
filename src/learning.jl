module Learning

export LGL, si_HITON_PC

using MultipleTesting
using Cauocc.Tests
using Cauocc.Misc
using Cauocc.Statfuns


function pw_univar_kernel!(X, data, stats, pvals, test_name, hps, levels, nz)
    adj_factor = nz ? 1 : 0
    
    if levels[X] - adj_factor < 2
        return
    end
    
    n_vars = size(data, 2)
    
    if nz
        sub_data = @view data[data[:, X] .!= 0, :]
    else
        sub_data = data
    end
        
    Ys = collect(X+1:n_vars)
    test_results = test(X, Ys, sub_data, test_name, hps, levels)
    
    for (Y, test_res) in zip(Ys, test_results)
        pair_index = sum(n_vars-1:-1:n_vars-X) - n_vars + Y
        stats[pair_index] = test_res.stat
        pvals[pair_index] = test_res.pval
    end   
end

function pw_univar_neighbors(data; test_name::String="mi", alpha::Float64=0.05, hps::Int=5, FDR::Bool=true, parallel="single")
    
    function nbr_kernel(X, data, test_name, hps, levels, nz, n_vars)        

        return zip(Ys, test_results)
    end
    
    n_vars = size(data, 2)
    n_pairs = convert(Int, n_vars * (n_vars - 1) / 2)
    levels = map(x -> length(unique(data[:, x])), 1:size(data, 2))
    
    nz = is_zero_adjusted(test_name)
    #adj_factor = nz ? 1 : 0
    
    if parallel == "single"
        pvals = zeros(Float64, n_pairs)
        stats = zeros(Float64, n_pairs)
        
        for X in 1:n_vars-1
            pw_univar_kernel!(X, data, stats, pvals, test_name, hps, levels, nz)
        end
        
    elseif startswith(parallel, "multi")
        pvals = SharedArray(Float64, n_pairs)
        stats = SharedArray(Float64, n_pairs)
        @sync @parallel for X in 1:n_vars-1
            for X in 1:n_vars-1
                pw_univar_kernel!(X, data, stats, pvals, test_name, hps, levels, nz)
            end
        end
        stats = stats.s
        pvals = pvals.s
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


function interleaving_phase(T::Int, candidates::Vector{Int}, data,
    test_name::String, max_k::Int, alpha::Float64, hps::Int=5, pwr::Float64=0.5, levels::Vector{Int}=Int[])
    TPC_dict = Dict{Int,Union{Tuple{Float64,Float64}, Void}}()
    
    if isempty(candidates)
        return TPC_dict
    end
    
    is_nz = is_zero_adjusted(test_name)
    
    TPC = [candidates[1]]
    TPC_dict[TPC[1]] = nothing
    
    for candidate in candidates[2:end]

        if is_nz
            sub_data = @view data[data[:, candidate] .!= 0, :]
        else
            sub_data = data
        end

        test_result = test_subsets(T, candidate, TPC, sub_data, test_name, max_k, alpha, hps=hps, pwr=pwr, levels=levels)
        
        if issig(test_result, alpha)
            push!(TPC, candidate)
            TPC_dict[candidate] = (test_result.stat, test_result.pval)
        end
    end
    
    TPC_dict
end


function elimination_phase(T::Int, TPC::Vector{Int}, data, test_name::String,
    max_k::Int, alpha::Float64, fast_elim::Bool, hps::Int=5, pwr::Float64=0.5, levels::Vector{Int}=Int[])
    
    PC_dict = Dict{Int,Union{Tuple{Float64,Float64}, Void}}()
    
    if isempty(TPC)
        return PC_dict
    end
    
    is_nz = is_zero_adjusted(test_name)
    
    PC = copy(TPC)
    for candidate in TPC
        PC_nocand = PC[PC .!= candidate]
        
        if is_nz
            sub_data = @view data[data[:, candidate] .!= 0, :]
        else
            sub_data = data
        end
        
        test_result = test_subsets(T, candidate, PC_nocand, sub_data, test_name, max_k, alpha, hps=hps, levels=levels)

        if !issig(test_result, alpha)
            deleteat!(PC, findin(PC, candidate))
        else
            PC_dict[candidate] = (test_result.stat, test_result.pval)
        end
    end

    PC_dict
end


function si_HITON_PC(target_var, data; test_name::String="mi", max_k::Int=3, alpha::Float64=0.05, hps::Int=5,
    pwr::Float64=0.5, FDR::Bool=true, fast_elim::Bool=true, weight_type::String="cond_logpval",
    univar_nbrs::Dict{Int,Tuple{Float64,Float64}}=Dict(), univar_step::Bool=true, debug::Int=0, header::Vector{String}=String[])
    #println("weight_type ", weight_type, " ", weight_type == "uni_stat")
    if typeof(target_var) == Int
        T = target_var
    elseif typeof(target_var) == String
        T = findfirst(header, target_var)   
    end
    
    if is_zero_adjusted(test_name)
        data = @view data[data[:, T] .!= 0, :]
    end
    
    test_variables = filter(x -> x != T, 1:size(data, 2))
    
    if debug > 0
        println("univariate")
    end
    
    # ...preprocessing and univariate testing code here...
    if isdiscrete(test_name)
        levels = map(x -> length(unique(data[:, x])), 1:size(data, 2))
        
         if levels[T] < 2
            return Dict{Union{Int, String},Float64}()
        end
        
        if univar_step
            univar_test_results = test(T, test_variables, data, test_name, hps, levels)
        end
    else
        levels = Int[]
           
        if univar_step
            univar_test_results = test(T, test_variables, data, test_name)
        end
    end
    
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
    

    if max_k > 0 
        candidate_pval_pairs = [(candidate, univar_nbrs[candidate][2])
                                for candidate in keys(univar_nbrs)]
        sort!(candidate_pval_pairs, by=x -> x[2])
        candidates = map(x -> x[1], candidate_pval_pairs)

        if debug > 0
            println("\tnumber of candidates:", length(candidates), candidates[1:min(length(candidates), 20)])
            println("\nINTERLEAVING\n")
        end

        TPC_dict = interleaving_phase(T, candidates, data, test_name, max_k, alpha, hps, pwr, levels)

        if isempty(TPC_dict)
            return Dict{Union{Int, String},Float64}()
        end

        # set test stats of the initial candidate to its univariate association results
        TPC_dict[candidates[1]] = univar_nbrs[candidates[1]]

        if debug > 0
            println("After interleaving:", length(TPC_dict))

            if debug > 1
                println(TPC_dict)
            end

            println("\nELIMINATION\n")
        end

        # elimination phase
        if length(TPC_dict) > 1
            PC_dict = elimination_phase(T, collect(keys(TPC_dict)), data, test_name, max_k, alpha, fast_elim, hps, pwr, levels)

            for nbr in keys(PC_dict)
                if PC_dict[nbr][1] == 0.0
                    PC_dict[nbr] = TPC_dict[nbr]
                end
            end  
        else
            PC_dict = Dict{Int,Union{Tuple{Float64,Float64}, Void}}([(nbr, univar_nbrs[nbr]) for nbr in keys(TPC_dict)])
        end

        if debug > 1
            println(PC_dict)
        end
    else
        PC_dict = univar_nbrs
    end
    
    # create weights
    nbr_dict = Dict{Union{Int, String},Float64}()
    weight_kind = String(split(weight_type, "_")[2])
    if startswith(weight_type, "uni")
        nbr_dict = Dict([(nbr, signed_weight(univar_nbrs[nbr]..., weight_kind)) for nbr in keys(PC_dict)])
    else
        nbr_dict = Dict([(nbr, signed_weight(PC_dict[nbr]..., weight_kind)) for nbr in keys(PC_dict)])
    end
    
    # translate variables if possible
    if !isempty(header)
        nbr_dict = Dict([(header[x], nbr_dict[x]) for x in keys(nbr_dict)])
    end
    
    nbr_dict
end


function LGL(data; test_name::String="mi", max_k::Int=3, alpha::Float64=0.05, hps::Int=5, pwr::Float64=0.5,
    FDR::Bool=true, global_univar::Bool=true, parallel::String="single", fast_elim::Bool=true, weight_type::String="cond_logpval",
        debug::Int=0, header::Vector{String}=String[], shared_stack::Channel=Channel(0))
    """
    parallel: 'single', 'multi_e', 'multi_il'
    """
    kwargs = Dict(:test_name => test_name, :max_k => max_k, :alpha => alpha, :hps => hps, :pwr => pwr, :FDR => FDR,
    :fast_elim => fast_elim, :weight_type => weight_type, :univar_step => !global_univar)
    
    if global_univar
        all_univar_nbrs = univar_neighbors(data; test_name=test_name, alpha=alpha, hps=hps, FDR=FDR, parallel=parallel)
        var_nbr_sizes = [(x, length(univar_neighbors[x])) for x in 1:size(data, 2)]
        target_vars = [nbr_size_pair[1] for nbr_size_pair in sort(var_nbr_sizes, by=x -> x[2])]
    else
        target_vars = 1:size(data, 2)
        all_univar_nbrs = Dict([(x, Dict{Int,Tuple{Float64,Float64}}()) for x in target_vars])
    end
    
    if parallel == "single" || nprocs() == 1
        nbr_dicts = map(x -> si_HITON_PC(x, data; univar_nbrs=all_univar_nbrs[x], kwargs...), target_vars)
    else
        # embarassingly parallel
        if parallel == "multi_ep"
            @sync nbr_dict_refs = [@spawn si_HITON_PC(x, data; univar_nbrs=all_univar_nbrs[x], kwargs...) for x in target_vars]
            nbr_dicts = map(fetch, nbr_dict_refs)
            
        # interleaved parallelism
        elseif parallel == "multi_il"
           
        else
            error("'$parallel' not a valid parallel mode")
        end
        
    end
    
    graph_dict = Dict([(target_var, nbr_dict) for (target_var, nbr_dict) in zip(target_vars, nbr_dicts)])
    
    if !isempty(header)
        graph_dict = Dict([(header[x], Dict([(header[y], graph_dict[x][y]) for y in keys(graph_dict[x])])) for x in keys(graph_dict)])
    end
    
    convert(Dict{Union{Int, String},Dict{Union{Int, String}, Float64}}, graph_dict)
end


end
