function prepare_lgl(data::AbstractMatrix{ElType}, test_name::String, time_limit::AbstractFloat,
    parallel::String,
    feed_forward::Bool, max_k::Integer, n_obs_min::Integer, hps::Integer, fast_elim::Bool, dense_cor::Bool,
    recursive_pcor::Bool, verbose::Bool, tmp_folder::AbstractString,
    edge_rule::AbstractString)  where {ElType<:Real}

    !isempty(tmp_folder) && @warn "tmp_folder currently not implemented"

    if edge_rule != "OR"
        @warn "edge_rule $(edge_rule) not a valid option, setting it to OR"
        edge_rule = "OR"
    end


    if time_limit == -1.0
        if parallel == "multi_il"
            time_limit = round(log2(size(data, 2)))
            verbose && println("Setting 'time_limit' to $time_limit s.")
        else
            time_limit = 0.0
        end
    end

    if time_limit != 0.0 && !endswith(parallel, "_il")
        @warn "Using time_limit without interleaved parallelism is not advised."
    elseif parallel == "multi_il" && time_limit == 0.0 && feed_forward
        @warn "Specify 'time_limit' when using interleaved parallelism to potentially increase speed."
    end

    disc_type = Int32
    cont_type = Float32

    if isdiscrete(test_name)
        verbose && println("Computing levels..")
        levels = get_levels(data)
        cor_mat = zeros(cont_type, 0, 0)
    else
        levels = disc_type[]

        if dense_cor && !is_zero_adjusted(test_name)
            data_dense = issparse(data) ? Matrix(data) : data
            cor_mat = convert(Matrix{cont_type}, cor(data_dense))
        else
            cor_mat = zeros(cont_type, 0, 0)
        end
    end


    if n_obs_min < 0# && is_zero_adjusted(test_name)
        if isdiscrete(test_name)#test_name == "mi_nz"
            max_levels = maximum(levels) - 1
            n_obs_min = hps * max_levels^(max_k+2) + 1
        else
            n_obs_min = 20
        end

        verbose && println("Automatically setting 'n_obs_min' to $n_obs_min for enhanced reliability.")
    end

    if n_obs_min > size(data, 1)
        error_msg = ""
        if max_k > 0
            error_msg *= ". Try using a smaller 'max_k' parameter (at the cost of higher numbers of indirect associations)."
        end

        error("Dataset has an insufficient number of observations, need at least $n_obs_min ('n_obs_min') for reliable tests$error_msg")
    end

    if verbose && is_zero_adjusted(test_name)
        n_unrel = sum(sum(data .!= 0, dims=1) .< n_obs_min)
        n_unrel > 0 && @warn "$n_unrel variables have insufficient observations (< $n_obs_min ('n_obs_min')) and will not be used for interaction prediction"
    end

    levels, cor_mat, time_limit, n_obs_min, fast_elim, disc_type, cont_type, tmp_folder, edge_rule
end


function prepare_univar_results(data::AbstractMatrix{ElType}, test_name::String, alpha::AbstractFloat, hps::Integer,
    n_obs_min::Integer, FDR::Bool, levels::Vector{DiscType}, parallel::String, cor_mat::AbstractMatrix{ContType},
    correct_reliable_only::Bool, verbose::Bool,
    tmp_folder::AbstractString="") where {ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}

    # precompute univariate associations and sort variables (fewest neighbors first)
    verbose && println("Computing univariate associations..")

    all_univar_nbrs = pw_univar_neighbors(data; test_name=test_name, alpha=alpha, hps=hps,
                                          n_obs_min=n_obs_min, FDR=FDR,
                                          levels=levels, parallel=parallel, workers_local=workers_all_local(),
                                          cor_mat=cor_mat, correct_reliable_only=correct_reliable_only,
                                          tmp_folder=tmp_folder)
    var_nbr_sizes = [(x, length(all_univar_nbrs[x])) for x in 1:size(data, 2)]
    target_vars = [nbr_size_pair[1] for nbr_size_pair in sort(var_nbr_sizes, by=x -> x[2])]

    if verbose
        println("\nUnivariate degree stats:")
        nbr_nums = map(length, values(all_univar_nbrs))
        println(summarystats(nbr_nums), "\n")
        if mean(nbr_nums) > size(data, 2) * 0.2
            warn_msg = ""
            if !is_zero_adjusted(test_name)
                warn_msg *= "Use 'heterogenenous=true' (if appropriate for your data) to increase speed. "
            end
            if iscontinuous(test_name)
                warn_msg *= "Consider setting 'sensitive=false'."
            end
            @warn "The univariate network is exceptionally dense, computations may be slow. $warn_msg"
        end
    end

    target_vars, all_univar_nbrs
end


function infer_conditional_neighbors(target_vars::Vector{Int}, data::AbstractMatrix{ElType},
     all_univar_nbrs::Dict{Int,NbrStatDict}, levels::Vector{DiscType},
     cor_mat::AbstractMatrix{ContType}, parallel::String, nonsparse_cond::Bool, recursive_pcor::Bool,
     cont_type::DataType, verbose::Bool, hiton_kwargs::Dict{Symbol, Any},
      interleaved_kwargs::Dict{Symbol, Any}) where {ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}

    # pre-allocate correlation matrix
    if recursive_pcor && is_zero_adjusted(hiton_kwargs[:test_name]) && iscontinuous(hiton_kwargs[:test_name])
        cor_mat = zeros(cont_type, size(data, 2), size(data, 2))
    end

    verbose && println("\nStarting conditioning search..")

    if nonsparse_cond && !endswith(parallel, "il")
        @warn "nonsparse_cond currently not implemented"
    end

    if parallel == "single"
        nbr_results = HitonState{Int}[si_HITON_PC(x, data, levels, cor_mat; univar_nbrs=all_univar_nbrs[x], hiton_kwargs...) for x in target_vars]
    else
        # embarassing parallelism
        if parallel == "multi_ep"
            nbr_results::Vector{HitonState{Int}} = @distributed (vcat) for x in target_vars
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

    rej_dict = Dict{Int, RejDict{Int}}()
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
                if !isempty(nbr_state.state_rejections)
                    rej_dict[target_var] = nbr_state.state_rejections
                end
            end
        end
    end

    nbr_dict, unfinished_state_dict, rej_dict
end


function LGL(data::AbstractMatrix; test_name::String="mi", max_k::Integer=3,
    alpha::AbstractFloat=0.01,
    hps::Integer=5, n_obs_min::Integer=-1, max_tests::Integer=Int(10e6),
    convergence_threshold::AbstractFloat=0.01,
    FDR::Bool=true, parallel::String="single", fast_elim::Bool=true, no_red_tests::Bool=true,
    weight_type::String="cond_stat", edge_rule::String="OR", nonsparse_cond::Bool=false,
    verbose::Bool=true, update_interval::AbstractFloat=30.0, edge_merge_fun=maxweight,
    tmp_folder::AbstractString="", debug::Integer=0, time_limit::AbstractFloat=-1.0,
    header=nothing, meta_variable_mask=nothing, dense_cor::Bool=true, recursive_pcor::Bool=true,
    cache_pcor::Bool=false, correct_reliable_only::Bool=true, feed_forward::Bool=true,
    track_rejections::Bool=false, all_univar_nbrs=nothing)
    """
    time_limit: -1.0 set heuristically, 0.0 no time_limit, otherwise time limit in seconds
    parallel: 'single', 'single_il', 'multi_ep', 'multi_il'
    fast_elim: currently always on
    """
    levels, cor_mat, time_limit, n_obs_min, fast_elim, disc_type, cont_type, tmp_folder, edge_rule =
                                                                              prepare_lgl(data, test_name,
                                                                                          time_limit, parallel,
                                                                                          feed_forward, max_k,
                                                                                          n_obs_min, hps,
                                                                                          fast_elim, dense_cor,
                                                                                          recursive_pcor, verbose,
                                                                                          tmp_folder,
                                                                                          edge_rule)

    hiton_kwargs = Dict(:test_name => test_name, :max_k => max_k, :alpha => alpha,
                        :hps => hps, :n_obs_min => n_obs_min, :max_tests => max_tests,
                        :fast_elim => fast_elim, :no_red_tests => no_red_tests, :FDR => FDR,
                        :weight_type => weight_type, :debug => debug, :time_limit => time_limit,
                        :track_rejections => track_rejections, :cache_pcor => cache_pcor)

    if all_univar_nbrs == nothing
        target_vars, all_univar_nbrs = prepare_univar_results(data, test_name, alpha, hps, n_obs_min,
                                                              FDR, levels, parallel, cor_mat,
                                                              correct_reliable_only, verbose,
                                                              tmp_folder)
    else
        target_vars = Vector{Int}(collect(keys(all_univar_nbrs)))
    end

    interleaved_kwargs = Dict(:update_interval => update_interval,
                              :convergence_threshold => convergence_threshold,
                              :feed_forward => feed_forward, :edge_rule => edge_rule,
                              :edge_merge_fun => edge_merge_fun,
                              :workers_local => workers_all_local(),
                              :variable_ids => header, :meta_variable_mask => meta_variable_mask)

    nbr_dict, unfinished_state_dict, rej_dict = learn_graph_structure(target_vars, data,
                                                                      all_univar_nbrs, levels,
                                                                      cor_mat, parallel,
                                                                      recursive_pcor,
                                                                      cont_type, time_limit, nonsparse_cond,
                                                                      verbose, track_rejections, hiton_kwargs,
                                                                      interleaved_kwargs)

    verbose && println("\nPostprocessing..")

    weights_dict = Dict{Int,Dict{Int,Float64}}()
    for target_var in keys(nbr_dict)
        weights_dict[target_var] = make_weights(nbr_dict[target_var], all_univar_nbrs[target_var],
                                                weight_type, test_name)
    end

    graph = make_symmetric_graph(weights_dict, edge_rule, edge_merge_fun=edge_merge_fun, max_var=size(data, 2),
                                 header=header)

    verbose && println("Complete.")

    LGLResult{Int}(graph, rej_dict, unfinished_state_dict)
end


"""
    learn_network(data_path::AbstractString, meta_data_path::AbstractString) -> FWResult{Int}

Works like learn_network(data::AbstractArray{ElType}), but takes file paths an OTU table and optionally a
meta data table as an input (instead of a data matrix).

- `data_path` - path to a file storing an OTU count table (and in the case of JLD2 possibly meta data)

- `meta_data_path` - optional path to a file with meta data

- `*_key`  - HDF5 keys to access data sets with OTU counts, Meta variables and variable names in a JLD2 file, if a data item is absent the corresponding key should be 'nothing'. More help under '?load_data'

- `verbose` - print progress information

- `transposed` - if `true`, rows of `data` are variables and columns are samples

- `kwargs...` - additional keyword arguments passed to learn_network(data::AbstractArray{ElType})


"""
function learn_network(data_path::AbstractString, meta_data_path=nothing;
    otu_data_key::AbstractString="otu_data",
    otu_header_key::AbstractString="otu_header", meta_data_key="meta_data",
    meta_header_key="meta_header", verbose::Bool=true,
    transposed::Bool=false, kwargs...)

    verbose && println("\n### Loading data ###\n")
    data, header, meta_data, meta_header = load_data(data_path, meta_data_path, otu_data_key=otu_data_key,
                                                     otu_header_key=otu_header_key, meta_data_key=meta_data_key,
                                                     meta_header_key=meta_header_key, transposed=transposed)


    if meta_data == nothing
        meta_mask = falses(length(header))
        check_data(data, header, meta_mask=meta_mask)
    else
        check_data(data, meta_data, header=header, meta_header=meta_header)
        data = hcat(data, meta_data)
        meta_mask = vcat(falses(length(header)), trues(length(meta_header)))
        header = vcat(header, meta_header)
    end

    learn_network(data; header=header, meta_mask=meta_mask, verbose=verbose, kwargs...)
end


"""
    learn_network(data::AbstractArray{<:Real}) -> FWResult{Int}

Learn an interaction network from a data table (including OTUs and optionally meta variables).

- `data` - data table with information on OTU counts and (optionally) meta variables

- `header` - names of variable columns in `data`

- `meta_mask` - true/false mask indicating which variables are meta variables

*Algorithmic parameters*:

- `heterogeneous` - enable heterogeneous mode for multi-habitat or -protocol data with at least thousands of samples (FlashWeaveHE)

- `sensitive` - enable fine-grained associations (FlashWeave-S, FlashWeaveHE-S),  `sensitive=false` results in the `fast` modes FlashWeave-F or FlashWeaveHE-F

- `max_k` - maximum size of conditioning sets, high values can strongly increase runtime. `max_k=0` results in no conditioning (univariate mode)

- `alpha` - threshold used to determine statistical significance

- `conv` - convergence threshold, i.e. if `conv=0.01` assume convergence if the number of edges increased by only 1% after 100% more runtime (checked in intervals)

- `feed_forward` - enable feed-forward heuristic

- `max_tests` - maximum number of conditional tests that should be performed on a variable pair before association is assumed

- `hps` - reliability criterion for statistical tests when `sensitive=false`

- `FDR` - perform False Discovery Rate correction (Benjamini-Hochberg method) on pairwise associations

- `n_obs_min` - don't compute associations between variables having less reliable samples (i.e. non-zero if `heterogeneous=true`) than this number. `-1`: automatically choose a threshold.

- `time_limit` - if feed-forward heuristic is active, determines the interval (seconds) at which neighborhood information is updated

*General parameters*:

- `normalize` - automatically choose and perform data normalization (based on `sensitive` and `heterogeneous`)

- `track_rejections` - store for each discarded edge, which variable set lead to its exclusion (can be memory intense for large networks)

- `verbose` - print progress information

- `transposed` - if `true`, rows of `data` are variables and columns are samples

- `prec` - precision in bits to use for calculations (16, 32, 64 or 128)

- `make_sparse` - use a sparse data representation (should be left at `true` in almost all cases)

- `make_onehot` - create one-hot encodings for meta data variables with more than two categories (should be left at `true` in almost all cases)

- `update_interval` - if `verbose=true`, determines the interval (seconds) at which network stat updates are printed

"""
function learn_network(data::AbstractMatrix; sensitive::Bool=true,
    heterogeneous::Bool=false, max_k::Integer=3, alpha::AbstractFloat=0.01,
    conv::AbstractFloat=0.01, header=nothing, meta_mask=nothing,
    feed_forward::Bool=true, normalize::Bool=true, track_rejections::Bool=false, verbose::Bool=true,
    transposed::Bool=false, prec::Integer=32, make_sparse::Bool=!sensitive || heterogeneous,
    make_onehot::Bool=true, max_tests=Int(10e6), hps::Integer=5, FDR::Bool=true, n_obs_min::Integer=-1,
    cache_pcor::Bool=false, time_limit::AbstractFloat=-1.0, update_interval::AbstractFloat=30.0)

    start_time = time()

    cont_mode = sensitive ? "fz" : "mi"
    het_mode = heterogeneous ? "_nz" : ""

    test_name = cont_mode * het_mode
    parallel_mode = nprocs() > 1 ? "multi_il" : "single_il"

    if transposed
        data = data'
    end

    if header == nothing
        header = ["X" * string(i) for i in 1:size(data, 2)]
    end

    if meta_mask == nothing
        meta_mask = falses(length(header))
    end

    check_data(data, header, meta_mask=meta_mask)

    n_mvs = sum(meta_mask)
    if verbose
        println("""Inferring network with $(mode_string(heterogeneous, sensitive, max_k))\n
        \tRun information:
        \tsensitive - $sensitive
        \theterogeneous - $heterogeneous
        \tmax_k - $(max_k)
        \talpha - $alpha
        \tsparse - $(issparse(data))
        \tworkers - $(length(workers()))
        \tOTUs - $(size(data, 2) - n_mvs)
        \tMVs - $(n_mvs)""")
    end


    params_dict = Dict(:test_name=>test_name, :parallel=>parallel_mode, :max_k=>max_k,
                       :alpha=>alpha, :convergence_threshold=>conv, :feed_forward=>feed_forward,
                       :track_rejections=>track_rejections, :verbose=>verbose,
                       :header=>header,
                       :max_tests=>max_tests, :hps=>hps, :FDR=>FDR, :n_obs_min=>n_obs_min,
                       :cache_pcor=>cache_pcor, :time_limit=>time_limit,
                       :update_interval=>update_interval)

    if normalize
       verbose && println("\n### Normalizing ###\n")
       input_data, header, meta_mask = normalize_data(data, test_name=test_name, header=header, meta_mask=meta_mask, prec=prec, verbose=verbose, make_sparse=make_sparse)
    else
       @warn "Skipping normalization, only experts should choose this option."
       input_data = data
    end


    verbose && println("\n### Learning interactions ###\n")
    lgl_results, time_taken = @timed LGL(input_data; params_dict...)

    # insert final parameters
    params_dict[:heterogeneous] = heterogeneous
    params_dict[:sensitive] = sensitive

    net_result = FWResult(lgl_results, header, meta_mask, params_dict)

    verbose && println("\nFinished inference. Total time taken: $(round(time_taken, digits=3))s")

    net_result
end
