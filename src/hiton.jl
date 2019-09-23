####################################
## Backend functions for Hiton-PC ##
####################################

function init_candidates(prev_accepted_dict::NbrStatDict, candidates::Vector{Int},
    candidates_unchecked::Vector{Int})

    if !isempty(prev_accepted_dict)
        accepted_dict = prev_accepted_dict
        candidates = candidates_unchecked
    else
        accepted_dict = NbrStatDict()
        candidates = candidates
    end

    candidates, accepted_dict
end


function candidate_in_blackwhite_lists!(candidate::Int, accepted::Vector{Int}, accepted_dict::NbrStatDict,
     whitelist::Set{Int}, blacklist::Set{Int}, debug::Integer)

    candidate_in_list = false
    if !isempty(whitelist) && candidate in whitelist
        push!(accepted, candidate)
        accepted_dict[candidate] = (NaN64, NaN64)

        debug > 0 && println("\tin whitelist")
        candidate_in_list = true
    end

    if !isempty(blacklist) && candidate in blacklist
        debug > 0 && println("\tin blacklist")
        candidate_in_list = true
    end

    candidate_in_list
end


function prepare_nzdata(T::Int, data::AbstractMatrix{ElType}, test_obj::AbstractTest) where {ElType<:Real}

    if needs_nz_view(T, data, test_obj)
        sub_data = @view data[data[:, T] .!= 0, :]
    else
        sub_data = data
    end

    sub_data
end


function update_sig_result!(test_result::TestResult, lowest_sig_Zs::Tuple{Vararg{Int64,N} where N<:Int}, candidate::Int,
     accepted::Vector{Int},
     accepted_dict::NbrStatDict, alpha::AbstractFloat, debug::Integer, rej_dict::RejDict{Int}, track_rejections::Bool,
     phase::Char, fast_elim::Bool, num_test_pair::Tuple{Int, Float64})

     if issig(test_result, alpha)
         push!(accepted, candidate)
         accepted_dict[candidate] = (test_result.stat, test_result.pval)

         debug > 0 && println("\taccepted: ", test_result)

     else
         if phase == 'E' && !fast_elim
            push!(accepted, candidate)
         end

         if track_rejections
             rej_dict[candidate] = (Tuple(lowest_sig_Zs), test_result, num_test_pair)
         end

         debug > 0 && println("\trejected: ", test_result, " through Z ", lowest_sig_Zs)
     end
 end

function check_candidate!(candidate::Int, T::Int, data::AbstractMatrix{ElType}, accepted::Vector{Int},
     accepted_dict::NbrStatDict,
    test_obj::AbstractTest, max_k::Integer, alpha::AbstractFloat, hps::Integer, n_obs_min::Integer, max_tests::Integer, debug::Integer, rej_dict::RejDict{Int},
    track_rejections::Bool, z::Vector{DiscType}, phase::Char, fast_elim::Bool)  where {ElType<:Real, DiscType<:Integer}

    data_prep = prepare_nzdata(candidate, data, test_obj)

    test_result, lowest_sig_Zs, num_tests, frac_tests = test_subsets(T, candidate, accepted, data_prep, test_obj,
                                                                     max_k, alpha, hps=hps, n_obs_min=n_obs_min,
                                                                     max_tests=max_tests, debug=debug, z=z)

    update_sig_result!(test_result, lowest_sig_Zs, candidate, accepted, accepted_dict, alpha, debug, rej_dict,
                             track_rejections, phase, fast_elim, (num_tests, frac_tests))
end

function hiton_backend(T::Int, candidates::AbstractVector{Int}, data::AbstractMatrix{ElType},
        test_obj::AbstractTest, max_k::Integer, alpha::AbstractFloat, hps::Integer=5, n_obs_min::Integer=0, max_tests::Integer=Int(1.5e9),
        prev_accepted_dict::NbrStatDict=Dict(),
        candidates_unchecked::Vector{Int}=Int[], time_limit::AbstractFloat=0.0, start_time::AbstractFloat=0.0,
        debug::Integer=0, whitelist::Set{Int}=Set{Int}(), blacklist::Set{Int}=Set{Int}(),
        rej_dict::RejDict{Int}=RejDict{Int}(), track_rejections::Bool=false,
        z::Vector{DiscType}=Int[], phase::Char='I'; fast_elim::Bool=true, no_red_tests::Bool=false) where {ElType<:Real, DiscType<:Integer}
    phase != 'I' && phase != 'E' && error("'phase' must be 'I' or 'E'")

    nz = is_zero_adjusted(test_obj)
    is_discrete = isdiscrete(test_obj)
    is_dense = !issparse(data)

    candidates, accepted_dict = init_candidates(prev_accepted_dict, candidates, candidates_unchecked)
    accepted = phase == 'E' ? copy(candidates) : Int[]

    for (cand_index, candidate) in enumerate(candidates)

        debug > 0 && println("\tTesting candidate $candidate ($cand_index out of $(length(candidates))) conditioned on $accepted, current set size: $(length(accepted))")

        candidate_in_list = candidate_in_blackwhite_lists!(candidate, accepted, accepted_dict, whitelist,
                                                           blacklist, debug)

        if !candidate_in_list
            if phase == 'E'
                deleteat!(accepted, findall(in(candidate), accepted))
            end

            check_candidate!(candidate, T, data, accepted, accepted_dict, test_obj, max_k, alpha, hps,
                             n_obs_min, max_tests, debug, rej_dict, track_rejections, z, phase, fast_elim)
        end

        if stop_reached(start_time, time_limit) && cand_index < length(candidates)
            candidates_unchecked = candidates[cand_index+1:end]
            return accepted_dict, candidates_unchecked
        end
    end
    accepted_dict, Int[]
end


function interleaving_phase(args...; add_initial_candidate::Bool=true,
    univar_nbrs::NbrStatDict=NbrStatDict())::Tuple{NbrStatDict,Vector{Int}}

    TPC_dict, candidates_unchecked = hiton_backend(args..., 'I')
    # set test stats of the initial candidate to its univariate association results
    if add_initial_candidate
        candidates = args[2]
        if haskey(TPC_dict, candidates[1])
            TPC_dict[candidates[1]] = univar_nbrs[candidates[1]]
        end
    end

    TPC_dict, candidates_unchecked
end

elimination_phase(args...; kwargs...)::Tuple{NbrStatDict,Vector{Int}} = hiton_backend(args..., 'E'; kwargs...)

function make_stopped_HitonState()
    phase = 'F'
    state_results = NbrStatDict()
    inter_results = NbrStatDict()
    unchecked_vars = Int[]
    state_rejections = RejDict{Int}()

    HitonState(phase, state_results, inter_results, unchecked_vars, state_rejections)
end

function init_hiton_pc(T::Int, data::AbstractMatrix{ElType}, test_name::String, levels::Vector{DiscType}, max_k::Integer,
     cor_mat::AbstractMatrix{ContType}, cache_pcor::Bool) where {ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}

    stop_hiton = false

    if isdiscrete(test_name)
        if isempty(levels)
            levels = get_levels(data)::Vector{DiscType}
        end

        if levels[T] < 2
            stop_hiton = true
            z = DiscType[]
        else
            z = !issparse(data) ? fill(-one(DiscType), size(data, 1)) : DiscType[]
        end
    else
        z = DiscType[]
    end

    test_obj = make_test_object(test_name, true, max_k=max_k, levels=levels, cor_mat=cor_mat, cache_pcor=cache_pcor)
    data_prep = prepare_nzdata(T, data, test_obj)
    test_variables = filter(x -> x != T, 1:size(data, 2))

    data_prep, levels, z, test_obj, test_variables, stop_hiton
end

function prepare_interleaving_phase(prev_state::HitonState{Int}, rej_dict::RejDict{Int},
     univar_nbrs::NbrStatDict, track_rejections::Bool)

    if prev_state.phase == 'I'
        prev_TPC_dict = prev_state.state_results
        candidates_unchecked = prev_state.unchecked_vars
        candidates = Int[]

        if track_rejections
            rej_dict = prev_state.state_rejections
        end
    else
        # sort candidates
        candidate_pval_pairs = Tuple{Int,Float64}[(candidate, univar_nbrs[candidate][2]) for candidate in keys(univar_nbrs)]
        sort!(candidate_pval_pairs, by=x -> x[2])
        candidates = map(x -> x[1], candidate_pval_pairs)
        candidates_unchecked = Int[]
        prev_TPC_dict = NbrStatDict()
    end
    candidates, candidates_unchecked, prev_TPC_dict, rej_dict
end


function prepare_elimination_phase(prev_state::HitonState{Int}, TPC_dict::NbrStatDict,
    rej_dict::RejDict{Int}, track_rejections::Bool, no_red_tests::Bool, fast_elim::Bool)

    if prev_state.phase == 'E'
        prev_PC_dict = prev_state.state_results

        if no_red_tests || fast_elim
            TPC_dict = prev_state.inter_results
        end

        PC_unchecked = prev_state.unchecked_vars
        PC_candidates = convert(Vector{Int}, [keys(prev_PC_dict)..., PC_unchecked...])

        if track_rejections
            rej_dict = prev_state.state_rejections
        end
    else
        prev_PC_dict = NbrStatDict()
        PC_unchecked = Int[]
        PC_candidates = convert(Vector{Int}, collect(keys(TPC_dict)))
    end

    PC_candidates, PC_unchecked, prev_PC_dict, TPC_dict, rej_dict
end


function update_PC_dict!(PC_dict::NbrStatDict, TPC_dict::NbrStatDict)

    for nbr in keys(PC_dict)
        if haskey(TPC_dict, nbr) && (TPC_dict[nbr][2] > PC_dict[nbr][2] || isnan(PC_dict[nbr][2]))
            PC_dict[nbr] = TPC_dict[nbr]
        end
    end
end


function make_final_HitonState(prev_state::HitonState{Int}, PC_dict::NbrStatDict,
     TPC_dict::NbrStatDict, rej_dict::RejDict{Int})

    # if previous state had converged, keep this information
    if prev_state.phase == 'C'
        phase = 'C'
        unchecked_vars = prev_state.unchecked_vars
        state_rejections = prev_state.state_rejections
    else
        phase = 'F'
        unchecked_vars = Int[]
        state_rejections = rej_dict
    end

    state_results = PC_dict
    inter_results = TPC_dict

    HitonState(phase, PC_dict, TPC_dict, unchecked_vars, state_rejections)
end

################################
## Main function for Hiton-PC ##
################################

function si_HITON_PC(T::Int, data::AbstractMatrix{ElType}, levels::Vector{DiscType}=DiscType[], cor_mat::Matrix{ContType}=zeros(ContType);
        test_name::String="mi", max_k::Int=3, alpha::Float64=0.01, hps::Int=5, n_obs_min::Int=0, max_tests::Int=Int(1.5e9),
        fast_elim::Bool=true, no_red_tests::Bool=false, FDR::Bool=true, weight_type::String="cond_stat",
        whitelist::Set{Int}=Set{Int}(), blacklist::Set{Int}=Set{Int}(),
        univar_nbrs::NbrStatDict=NbrStatDict(),
        prev_state::HitonState{Int}=HitonState{Int}('S', OrderedDict(), OrderedDict(), [], Dict()),
        debug::Int=0, time_limit::Float64=0.0, track_rejections::Bool=false,
         cache_pcor::Bool=true) where {ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}

    debug > 0 && println("Finding neighbors for $T")

    rej_dict = RejDict{Int}()

    data_prep, levels, z, test_obj, test_variables, stop_hiton = init_hiton_pc(T, data, test_name, levels, max_k, cor_mat, cache_pcor)

    if stop_hiton
        return make_stopped_HitonState()
    end

    start_time = time_limit > 0.0 ? time() : 0.0

    if debug > 0
        println("UNIVARIATE")
        uni_printinf = collect(zip(test_variables, univar_nbrs))
        println("\tFirst up to 200 neighbors:", uni_printinf[1:min(200, length(uni_printinf))])
    end

    # if conditioning should be performed
    if max_k > 0
        # if the global network has converged
        if prev_state.phase == 'C'
            if !isempty(prev_state.inter_results)
                TPC_dict = prev_state.inter_results
                PC_dict = prev_state.state_results
            else
                TPC_dict = NbrStatDict()
                PC_dict = NbrStatDict()
            end
        else
            TPC_dict = NbrStatDict()
            PC_dict = NbrStatDict()

            if prev_state.phase == 'I' || prev_state.phase == 'S'

                candidates, candidates_unchecked, prev_TPC_dict, rej_dict = prepare_interleaving_phase(prev_state, rej_dict, univar_nbrs, track_rejections)

                if debug > 0
                    println("\tnumber of candidates:", length(candidates), candidates[1:min(length(candidates), 20)])
                    println("\nINTERLEAVING\n")
                end

                if isempty(candidates)
                    return make_stopped_HitonState()
                end

                TPC_dict, candidates_unchecked = interleaving_phase(T, candidates, data_prep, test_obj, max_k,
                                                                    alpha, hps, n_obs_min, max_tests,
                                                                    prev_TPC_dict, candidates_unchecked,
                                                                    time_limit, start_time, debug, whitelist,
                                                                    blacklist, rej_dict, track_rejections, z,
                                                                    add_initial_candidate=prev_state.phase=='S',
                                                                    univar_nbrs=univar_nbrs)

                if !isempty(candidates_unchecked)

                    debug > 0 && println("Time limit exceeded, reporting incomplete results")

                    return HitonState('I', TPC_dict, NbrStatDict(), candidates_unchecked, rej_dict)
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
            PC_candidates, PC_unchecked, prev_PC_dict, TPC_dict, rej_dict = prepare_elimination_phase(prev_state, TPC_dict, rej_dict, track_rejections,
                                                                                                      no_red_tests, fast_elim)


            PC_dict, TPC_unchecked = elimination_phase(T, PC_candidates, data_prep, test_obj, max_k, alpha,
                                                       hps, n_obs_min, max_tests, prev_PC_dict, PC_unchecked, time_limit,
                                                        start_time, debug, whitelist, blacklist, rej_dict,
                                                        track_rejections, z, fast_elim=fast_elim,
                                                         no_red_tests=no_red_tests)

            if !isempty(TPC_unchecked)
                debug > 0 && println("Time limit exceeded, reporting incomplete results")

                return HitonState('E', PC_dict, TPC_dict, TPC_unchecked, rej_dict)
            end
        end

        # if redundant tests were skipped in elimination phase, check
        # if lower weights were previously found during interleaving phase
        if no_red_tests || fast_elim
            update_PC_dict!(PC_dict, TPC_dict)
        end

        debug > 1 && println(PC_dict)

    else
        PC_dict = univar_nbrs
    end

    return make_final_HitonState(prev_state, PC_dict, TPC_dict, rej_dict)
end
