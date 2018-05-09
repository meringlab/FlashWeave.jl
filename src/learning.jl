module Learning

export LGL, learn_network

using DataStructures
using StatsBase

using FlashWeave.Tests
using FlashWeave.Types
using FlashWeave.Misc
using FlashWeave.Hiton
using FlashWeave.Interleaved
using FlashWeave.Preclustering


function prepare_lgl(data::AbstractMatrix{ElType}, test_name::String, time_limit::AbstractFloat, parallel::String,
    feed_forward::Bool, max_k::Integer, n_obs_min::Integer, hps::Integer, fast_elim::Bool,
     recursive_pcor::Bool, verbose::Bool)  where {ElType<:Real}
    if time_limit == -1.0
        if parallel == "multi_il"
            time_limit = round(log2(size(data, 2)))
            println("Setting 'time_limit' to $time_limit s.")
        else
            time_limit = 0.0
        end
    end

    if time_limit != 0.0 && parallel != "multi_il"
        warn("Using time_limit without interleaved parallelism is not advised.")
    elseif parallel == "multi_il" && time_limit == 0.0 && feed_forward
        warn("Specify 'time_limit' when using interleaved parallelism to increase speed.")
    end

    if time_limit != 0.0 && !fast_elim
        warn("Setting fast_elim to true because time_limit has been specified")
        fast_elim = true
    end

    disc_type = Int32
    cont_type = Float32

    if isdiscrete(test_name)
        if verbose
            println("Computing levels..")
        end
        levels = get_levels(data)
        cor_mat = zeros(cont_type, 0, 0)
    else
        levels = disc_type[]

        if recursive_pcor && !is_zero_adjusted(test_name)
            cor_mat = convert(Matrix{cont_type}, cor(data))
        else
            cor_mat = zeros(cont_type, 0, 0)
        end
    end


    if n_obs_min < 0 && is_zero_adjusted(test_name)
        if test_name == "mi_nz"
            max_levels = maximum(levels) - 1
            n_obs_min = hps * max_levels^(max_k+2) + 1
        else
            n_obs_min = 20
        end

        if verbose
            println("Automatically setting 'n_obs_min' to $n_obs_min to enhance reliability.")
        end
    end

    levels, cor_mat, time_limit, n_obs_min, fast_elim, disc_type, cont_type
end


function prepare_univar_results(data::AbstractMatrix{ElType}, test_name::String, alpha::AbstractFloat, hps::Integer,
    n_obs_min::Integer, FDR::Bool, levels::Vector{DiscType}, parallel::String, cor_mat::AbstractMatrix{ContType},
    correct_reliable_only::Bool, verbose::Bool) where {ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}

    # precompute univariate associations and sort variables (fewest neighbors first)
    if verbose
        println("Computing univariate associations..")
        tic()
    end

    all_univar_nbrs = pw_univar_neighbors(data; test_name=test_name, alpha=alpha, hps=hps, n_obs_min=n_obs_min, FDR=FDR,
                                          levels=levels, parallel=parallel, workers_local=workers_all_local(),
                                          cor_mat=cor_mat, correct_reliable_only=correct_reliable_only)
    var_nbr_sizes = [(x, length(all_univar_nbrs[x])) for x in 1:size(data, 2)]
    target_vars = [nbr_size_pair[1] for nbr_size_pair in sort(var_nbr_sizes, by=x -> x[2])]

    if verbose
        println("\nUnivariate degree stats:")
        nbr_nums = map(length, values(all_univar_nbrs))
        println(summarystats(nbr_nums), "\n")
        if mean(nbr_nums) > size(data, 2) * 0.2
            warn("The univariate network is exceptionally dense, computations may be very slow.
                 Check if appropriate normalization was used (employ niche-mode if not yet the case)
                 and try using the AND rule to gain speed.")
        end
        toc()
    end

    target_vars, all_univar_nbrs
end


function make_preclustering(precluster_sim::AbstractFloat, data::AbstractMatrix{ElType}, target_vars::Vector{Int},
    cor_mat::AbstractMatrix{ContType}, levels::Vector{DiscType}, test_name::String,
     all_univar_nbrs::Dict{Int,NbrStatDict},
    cluster_mode::String, verbose::Bool) where {ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}

    if precluster_sim != 0.0
        if verbose
            println("Clustering..")
            tic()
        end

        univar_matrix = pw_unistat_matrix(data, test_name; pw_stat_dict=all_univar_nbrs)
        clust_repres, clust_dict = cluster_data(data, test_name, cluster_sim_threshold=precluster_sim,
                                    sim_mat=univar_matrix, greedy=cluster_mode == "greedy")

        if verbose
            println("\tfound $(length(clust_repres)) clusters")
            toc()
        end

        data = data[:, clust_repres]

        if !isempty(cor_mat)
            cor_mat = cor_mat[clust_repres, clust_repres]
        end

        target_vars = collect(1:length(clust_repres))
        var_clust_dict = Dict{Int,Int}(zip(clust_repres, 1:length(clust_repres)))
        clust_var_dict = Dict{Int,Int}(zip(1:length(clust_repres), clust_repres))
        all_univar_nbrs = map_edge_keys(all_univar_nbrs, var_clust_dict)

        if isdiscrete(test_name)
            levels = levels[clust_repres]
        end
    else
        clust_dict = Dict{Int,Int}()
        clust_var_dict = Dict{Int,Int}()
    end

    target_vars, all_univar_nbrs, clust_dict, clust_var_dict, levels
end


function infer_conditional_neighbors(target_vars::Vector{Int}, data::AbstractMatrix{ElType},
     all_univar_nbrs::Dict{Int,NbrStatDict}, levels::Vector{DiscType},
     cor_mat::AbstractMatrix{ContType}, parallel::String, nonsparse_cond::Bool, recursive_pcor::Bool,
     cont_type::DataType, verbose::Bool, hiton_kwargs::Dict{Symbol, Any},
      interleaved_kwargs::Dict{Symbol, Any}) where {ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}

    # pre-allocate correlation matrix
    if recursive_pcor && is_zero_adjusted(hiton_kwargs[:test_name])
        cor_mat = zeros(cont_type, size(data, 2), size(data, 2))
    end

    if verbose
        println("Running si_HITON_PC for each variable..")
        tic()
    end

    if nonsparse_cond && !endswith(parallel, "il")
        data = full(data)
    end

    if parallel == "single"
        nbr_results = HitonState{Int}[si_HITON_PC(x, data, levels, cor_mat; univar_nbrs=all_univar_nbrs[x], hiton_kwargs...) for x in target_vars]
    else
        # embarassing parallelism
        if parallel == "multi_ep"
            nbr_results::Vector{HitonState{Int}} = @parallel (vcat) for x in target_vars
                si_HITON_PC(x, data, levels, cor_mat; univar_nbrs=all_univar_nbrs[x], hiton_kwargs...)
            end

        # interleaved mode
        elseif endswith(parallel, "il")
            il_dict = interleaved_backend(target_vars, data, all_univar_nbrs, levels, cor_mat, hiton_kwargs; parallel=parallel,
                                          nonsparse_cond=nonsparse_cond, verbose=verbose, interleaved_kwargs...)
            nbr_results = HitonState{Int}[il_dict[target_var] for target_var in target_vars]
        else
            error("'$parallel' not a valid parallel mode")
        end

    end

    Dict{Int,HitonState{Int}}(zip(target_vars, nbr_results))
end


function learn_graph_structure(target_vars::Vector{Int}, data::AbstractMatrix{ElType},
    all_univar_nbrs::Dict{Int,NbrStatDict},
    levels::Vector{DiscType}, cor_mat::AbstractMatrix{ContType}, parallel::String, recursive_pcor::Bool,
    cont_type::DataType, time_limit::AbstractFloat, nonsparse_cond::Bool, verbose::Bool, track_rejections::Bool,
    hiton_kwargs::Dict{Symbol, Any}, interleaved_kwargs::Dict{Symbol, Any}) where {ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}

    rej_dict = Dict{Int, RejDict{Int}}()#Dict{Int, Dict{Int, Tuple{Tuple,TestResult}}}()
    unfinished_state_dict = Dict{Int, HitonState{Int}}()

    # if only univar network should be learned, dont do anything
    if hiton_kwargs[:max_k] == 0
        nbr_dict = all_univar_nbrs

    # else, run full conditioning
    else
        nbr_results_dict = infer_conditional_neighbors(target_vars, data, all_univar_nbrs, levels, cor_mat, parallel,
                                                       nonsparse_cond, recursive_pcor, cont_type, verbose, hiton_kwargs,
                                                       interleaved_kwargs)

        nbr_dict = Dict{Int,NbrStatDict}([(target_var, nbr_state.state_results) for (target_var, nbr_state) in nbr_results_dict])

        if time_limit != 0.0 || interleaved_kwargs[:convergence_threshold] != 0.0
            for (target_var, nbr_state) in nbr_results_dict
                if !isempty(nbr_state.unchecked_vars)
                    unfinished_state_dict[target_var] = nbr_state
                end
            end
        end

        if track_rejections
            for (target_var, nbr_state) in nbr_results_dict
                rej_dict[target_var] = nbr_state.state_rejections
            end
        end

        if verbose
            toc()
        end
    end

    nbr_dict, unfinished_state_dict, rej_dict
end


function map_clusters_to_variables(nbr_dict::Dict{Int,NbrStatDict},
     all_univar_nbrs::Dict{Int,NbrStatDict},
     rej_dict::Dict{Int,Tuple{Tuple,TestResult}}, clust_var_dict::Dict{Int,Int}, track_rejections::Bool)

    nbr_dict = map_edge_keys(nbr_dict, clust_var_dict)
    all_univar_nbrs = map_edge_keys(all_univar_nbrs, clust_var_dict)

    if track_rejections
        rej_dict = map_edge_keys(rej_dict, clust_var_dict)
    end

    nbr_dict, all_univar_nbrs, rej_dict
end


function LGL(data::AbstractMatrix{ElType}; test_name::String="mi", max_k::Integer=3, alpha::AbstractFloat=0.01,
    hps::Integer=5, n_obs_min::Integer=-1, max_tests::Integer=Int(1.5e6), convergence_threshold::AbstractFloat=0.01, FDR::Bool=true,
    parallel::String="single", fast_elim::Bool=true, no_red_tests::Bool=true, precluster_sim::AbstractFloat=0.0,
    weight_type::String="cond_stat", edge_rule::String="OR", nonsparse_cond::Bool=false,
    verbose::Bool=true, update_interval::AbstractFloat=30.0, output_folder::String="", output_interval::Real=update_interval*10, edge_merge_fun=maxweight,
    debug::Integer=0, time_limit::AbstractFloat=-1.0, header::AbstractVector{String}=String[],
    recursive_pcor::Bool=true, cache_pcor::Bool=false, correct_reliable_only::Bool=true, feed_forward::Bool=true,
    track_rejections::Bool=false, cluster_mode::AbstractString="greedy") where {ElType<:Real}
    """
    time_limit: -1.0 set heuristically, 0.0 no time_limit, otherwise time limit in seconds
    parallel: 'single', 'single_il', 'multi_ep', 'multi_il'
    fast_elim: currently always on
    """
    levels, cor_mat, time_limit, n_obs_min, fast_elim, disc_type, cont_type = prepare_lgl(data, test_name, time_limit, parallel,
                                                                                          feed_forward, max_k, n_obs_min, hps, fast_elim,
                                                                                          recursive_pcor, verbose)

    hiton_kwargs = Dict(:test_name => test_name, :max_k => max_k, :alpha => alpha, :hps => hps, :n_obs_min => n_obs_min, :max_tests => max_tests,
                  :fast_elim => fast_elim, :no_red_tests => no_red_tests, :FDR => FDR,
                  :weight_type => weight_type, :debug => debug,
                  :time_limit => time_limit, :track_rejections => track_rejections, :cache_pcor => cache_pcor)

    target_vars, all_univar_nbrs = prepare_univar_results(data, test_name, alpha, hps, n_obs_min, FDR, levels,
                                                          parallel, cor_mat, correct_reliable_only, verbose)

    target_vars, all_univar_nbrs, clust_dict, clust_var_dict, levels = make_preclustering(precluster_sim, data, target_vars, cor_mat,
                                                                                          levels, test_name, all_univar_nbrs, cluster_mode, verbose)

    interleaved_kwargs = Dict(:update_interval => update_interval, :convergence_threshold => convergence_threshold,
                                  :feed_forward => feed_forward, :edge_rule => edge_rule, :edge_merge_fun => edge_merge_fun,
                                  :workers_local => workers_all_local(), :output_folder => output_folder, :output_interval => output_interval)

    nbr_dict, unfinished_state_dict, rej_dict = learn_graph_structure(target_vars, data, all_univar_nbrs, levels, cor_mat, parallel,
                                                                      recursive_pcor,
                                                                      cont_type, time_limit, nonsparse_cond,
                                                                      verbose, track_rejections, hiton_kwargs,
                                                                       interleaved_kwargs)

    if verbose
        println("Postprocessing..")
        tic()
    end

    if precluster_sim != 0.0
        nbr_dict, all_univar_nbrs, rej_dict = map_clusters_to_variables(nbr_dict, all_univar_nbrs, rej_dict, clust_var_dict, track_rejections)
    end

    weights_dict = Dict{Int,Dict{Int,Float64}}()
    for target_var in keys(nbr_dict)
        weights_dict[target_var] = make_weights(nbr_dict[target_var], all_univar_nbrs[target_var], weight_type, test_name)
    end

    graph = make_symmetric_graph(weights_dict, edge_rule, edge_merge_fun=edge_merge_fun, max_var=size(data, 2))

    if verbose
        println("Complete.")
        toc()
    end

    LGLResult{Int}(graph, rej_dict, unfinished_state_dict)
end


function learn_network(data::AbstractArray{ElType}; sensitive::Bool=true, heterogeneous::Bool=false,
                       maxk::Integer=3, alpha::AbstractFloat=0.01, hps::Integer=5,
                       normalize_data::Bool=true, verbose::Bool=true, lgl_kwargs...) where {ElType<:Real}

    start_time = time()

    cont_mode = sensitive ? "fz" : "mi"
    het_mode = heterogeneous ? "_nz" : ""

    test_name = cont_mode * het_mode
    parallel_mode = nprocs() > 1 ? "multi_il" : "single_il"


    if normalize_data
        normalize!(data)
    end

    if verbose
        println("""Inferring network\n
        \tSettings:
        \t\tsensitive - $sensitive
        \t\theterogeneous - $heterogeneous
        \t\tmax_k - $(max_k)
        \t\talpha - $alpha
        \t\tsparse - $(issparse(data))
        \t\tworkers - $(nprocs())""")
    end

    params_dict = Dict(:test_name=>test_name, :max_k=>max_k, :alpha=>alpha, :hps=>hps,
                     :parallel=>parallel_mode)
    merge!(params_dict, lgl_kwargs)

    lgl_results = LGL(data; params_dict...)

    time_taken = time() - start_time()

    if verbose
        println("Finished inference. Time taken: ", time_taken, "s")
    end

    stats_dict = Dict(:time_taken=>time_taken, :converged=>!isempty(lgl_results.unfinished_states))
    meta_dict = Dict("params"=>params_dict, "stats"=>stats_dict)
    lgl_results, meta_dict
end

end
