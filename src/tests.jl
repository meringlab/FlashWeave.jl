function issig(test_res::TestResult, alpha::AbstractFloat; test_obj=nothing)
    # special case for fz_nz (returns NaN pvalues if any Z has zero variance, such cases should not lead to rejection)
    if isa(test_obj, FzTestCond) && is_zero_adjusted(test_obj) && isnan(test_res.pval)
        true
    else
        test_res.pval < alpha && test_res.suff_power == true
    end
end

sufficient_power(levels_x::Integer, levels_y::Integer, n_obs::Integer, hps::Integer) = (n_obs / (levels_x * levels_y)) > hps
sufficient_power(levels_x::Integer, levels_y::Integer, levels_z::Integer, n_obs::Integer, hps::Integer) = (n_obs / (levels_x * levels_y * levels_z)) > hps

"""Can't be used for MiTestCond since levels_z requires contingency table"""
function sufficient_power(X, Y, data::AbstractMatrix{<:Real}, test_obj::Union{MiTest,FzTest,FzTestCond}, n_obs_min::Integer, hps::Integer=5)
    n_obs = size(data, 1)
    n_obs < n_obs_min && return false

    if isdiscrete(test_obj)
        @inbounds levels_x = test_obj.levels[X]
        @inbounds levels_y = test_obj.levels[Y]
        offset_x, offset_y = offset_levels(levels_x, levels_y)
        !sufficient_power(levels_x-offset_x, levels_y-offset_y, n_obs, hps) && return false
    end
    true
end

##################
### UNIVARIATE ###
##################

### discrete

function test(X::Int, Y::Int, data::AbstractMatrix{<:Integer}, test_obj::AbstractContTest, hps::Integer, n_obs_min::Int=0)
    @inbounds begin
        levels_x = test_obj.levels[X]
        levels_y = test_obj.levels[Y]
        max_val_x = test_obj.max_vals[X]
        max_val_y = test_obj.max_vals[Y]
    end

    if !sufficient_power(X, Y, data, test_obj, n_obs_min, hps)
        mi_stat = 0.0
        df = 0
        pval = 1.0
        suff_power = false
    else
        if !issparse(data)
            contingency_table!(X, Y, data, test_obj.ctab)
        else
            contingency_table!(X, Y, data, test_obj)
        end

        if is_zero_adjusted(test_obj)
            sub_ctab = nz_adjust_cont_tab(max_val_x, max_val_y, test_obj.ctab)
            levels_x = size(sub_ctab, 1)
            levels_y = size(sub_ctab, 2)
        else
            sub_ctab = test_obj.ctab
        end

        n_obs = sum(sub_ctab)

        if n_obs < n_obs_min || !sufficient_power(levels_x, levels_y, n_obs, hps)
            mi_stat = 0.0
            df = 0
            pval = 1.0
            suff_power = false
        else
            mi_stat = mutual_information(sub_ctab, levels_x, levels_y, test_obj.marg_i, test_obj.marg_j)

            df = adjust_df(test_obj.marg_i, test_obj.marg_j, levels_x, levels_y)
            pval = mi_pval(abs(mi_stat), df, n_obs)
            suff_power = true
        end
    end

    TestResult(mi_stat, pval, df, suff_power)
end


function test(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{<:Integer},
        test_obj::AbstractContTest, hps::Integer, n_obs_min::Int=0)
    """CRITICAL: expects zeros to be trimmed from X if nz_test
    is provided!

    Test all variables Ys for univariate association with X"""
    @inbounds levels_x = test_obj.levels[X]
    if levels_x < 2
        return TestResult[TestResult(0.0, 1.0, 0, false) for Y in Ys]
    else
        return map(Y -> test(X, Y, data, test_obj, hps, n_obs_min), Ys)::Vector{TestResult}
    end
end

# convenience wrapper
function test(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{<:Integer},
        test_name::String, hps::Integer=5, n_obs_min::Int=0, levels::Vector{<:Integer}=Int[], max_vals::Vector{<:Integer}=Int[])
    if isempty(levels) || isempty(max_vals)
        levels = get_levels(data)
        max_vals = get_max_vals(data)
    end
    test_obj = make_test_object(test_name, false, max_k=0, levels=levels, max_vals=max_vals, cor_mat=zeros(Float64, 0, 0))
    test(X, Ys, data, test_obj, hps, n_obs_min)
end


### continuous

function test(X::Int, Y::Int, data::AbstractMatrix{<:Real}, test_obj::FzTest,
        n_obs_min::Integer=0, Y_adjusted::Bool=false)

    if !sufficient_power(X, Y, data, test_obj, n_obs_min)
        p_stat = 0.0
        df = 0
        pval = 1.0
        n_obs = 0
    else
        nz = is_zero_adjusted(test_obj)

        @inbounds if isempty(test_obj.cor_mat) || nz
            if issparse(data)
                p_stat, n_obs = cor(X, Y, data, nz)

                if n_obs < n_obs_min
                    p_stat = 0.0
                end
            else
                if nz && !Y_adjusted
                    sub_data = @view data[data[:, Y] .!= 0, :]
                else
                    sub_data = data
                end

                if isempty(sub_data)
                    p_stat = 0.0
                    n_obs = 0
                else
                    n_obs = size(sub_data, 1)

                    if n_obs >= n_obs_min
                        sub_x_vec = @view sub_data[:, X]
                        sub_y_vec = @view sub_data[:, Y]

                        p_stat = cor(sub_x_vec, sub_y_vec)
                    else
                        p_stat = 0.0
                    end
                end
            end
        else
            n_obs = size(data, 1)
            p_stat = n_obs >= n_obs_min ? test_obj.cor_mat[X, Y] : 0.0
        end

        df = 0
        pval = fz_pval(p_stat, n_obs, 0)
    end

    TestResult(p_stat, pval, df, n_obs >= n_obs_min)
end


function test(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{<:AbstractFloat},
        test_obj::AbstractCorTest, n_obs_min::Integer=0)
    """CRITICAL: expects zeros to be trimmed from X if nz_test
    is provided!

    Test all variables Ys for univariate association with X"""

    map(Y -> test(X, Y, data, test_obj, n_obs_min, false), Ys)::Vector{TestResult}
end

# convenience wrapper
function test(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{<:AbstractFloat},
        test_name::String, n_obs_min::Integer=0)
    test_obj = make_test_object(test_name, false, max_k=0, levels=Int[], max_vals=Int[], cor_mat=zeros(Float64, 0, 0))
    test(X, Ys, data, test_obj, n_obs_min)
end

###################
### CONDITIONAL ###
###################

### discrete

function test(X::Int, Y::Int, Zs::NTuple{N,T} where {N,T<:Integer}, data::AbstractMatrix{<:Integer}, test_obj::MiTestCond, hps::Integer, z::AbstractVector{<:Integer}=Int[])
    """Test association between X and Y"""
    @inbounds begin
        levels_x = test_obj.levels[X]
        levels_y = test_obj.levels[Y]
        max_val_x = test_obj.max_vals[X]
        max_val_y = test_obj.max_vals[Y]
    end

    if !issparse(data)
        levels_z = contingency_table!(X, Y, Zs, data, test_obj.ctab, z, test_obj.zmap.cum_levels, test_obj.zmap.z_map_arr)
    else
        contingency_table!(X, Y, Zs, data, test_obj)
        levels_z = test_obj.zmap.levels_total
    end

    if is_zero_adjusted(test_obj)
        sub_ctab = nz_adjust_cont_tab(max_val_x, max_val_y, test_obj.ctab)
        levels_x = size(sub_ctab, 1)
        levels_y = size(sub_ctab, 2)
    else
        sub_ctab = test_obj.ctab
    end

    n_obs = sum(sub_ctab)

    if !sufficient_power(levels_x, levels_y, levels_z, n_obs, hps)
        mi_stat = 0.0
        df = 0
        pval = 1.0
        suff_power = false
    else
        mi_stat = mutual_information(sub_ctab, levels_x, levels_y, levels_z, test_obj.marg_i, test_obj.marg_j,
                                     test_obj.marg_k)

        df = adjust_df(test_obj.marg_i, test_obj.marg_j, levels_x, levels_y, levels_z)
        pval = mi_pval(abs(mi_stat), df, n_obs)
        suff_power = true
    end

    TestResult(mi_stat, pval, df, suff_power)
end

# convenience wrapper
function test(X::Int, Y::Int, Zs::NTuple{N,T} where {N,T<:Integer}, data::AbstractMatrix{<:Integer}, test_name::String, hps::Integer=5, 
    levels::Vector{<:Integer}=Int[], max_vals::Vector{<:Integer}=Int[])
    """Convenience function for module tests"""
    if isempty(levels) || isempty(max_vals)
        levels = get_levels(data)
        max_vals = get_max_vals(data)
    end

    test_obj = MiTestCond(levels, is_zero_adjusted(test_name) ? Nz() : NoNz(), length(Zs), max_vals)

    z = issparse(data) ? eltype(levels)[] : zeros(eltype(levels), size(data, 1))
    test(X, Y, Zs, data, test_obj, hps, z)
end



### continuous

function test(X::Int, Y::Int, Zs::NTuple{N,T} where {N,T<:Integer}, data::AbstractMatrix{<:AbstractFloat},
    test_obj::FzTestCond, n_obs_min::Integer)
    """Critical: expects zeros to be trimmed from both X and Y if nz is true"""

    if sufficient_power(X, Y, data, test_obj, n_obs_min)
        p_stat = isempty(test_obj.cor_mat) ? pcor(X, Y, Zs, data) : pcor_rec(X, Y, Zs, test_obj.cor_mat, test_obj.pcor_set_dict, test_obj.cache_pcor)
        pval = fz_pval(p_stat, size(data, 1), 0)
        suff_power = true
    else
        p_stat = 0.0
        pval = 1.0
        suff_power = false
    end

    TestResult(p_stat, pval, 0, suff_power)
end

# convenience wrapper
function test(X::Int, Y::Int, Zs::NTuple{N,T} where {N,T<:Integer}, data::AbstractMatrix{<:AbstractFloat}, test_name::String;
     recursive::Bool=true, n_obs_min::Integer=0)
    """Convenience function for module tests"""
    cor_mat = recursive ? cor(data) : zeros(Float64, 0, 0)
    test_obj = FzTestCond(cor_mat, Dict{String,Dict{String,eltype(cor_mat)}}(), is_zero_adjusted(test_name) ? Nz() : NoNz(),
        true)
    test(X, Y, Zs, data, test_obj, n_obs_min)
end


## MAIN SUBSET TEST FUNCTION ##

function test_subsets(X::Int, Y::Int, Z_total::AbstractVector{Int}, data::AbstractMatrix{<:Real},
    test_obj::AbstractTest, max_k::Integer, alpha::AbstractFloat; hps::Integer=5, n_obs_min::Integer=0, max_tests::Integer=Int(1.5e9),
    debug::Int=0, Z_wanted::AbstractVector{Int}=Int[], z::Vector{<:Integer}=Int[])

    isempty(Z_total) && return TestResult(NaN, NaN, -1, true), (-1,), -1, NaN

    lowest_sig_result = TestResult(0.0, 0.0, 0.0, true)
    lowest_sig_Zs = ()
    discrete_test = isdiscrete(test_obj)
    num_tests = 0
    nz = is_zero_adjusted(test_obj)

    if !discrete_test && nz
        if n_obs_min > size(data, 1)
            return TestResult(0.0, 1.0, 0.0, false), lowest_sig_Zs, num_tests, 0.0
        end

        # reset cached partial correlations since they depend on X and Y
        if test_obj.cache_pcor
            empty!(test_obj.pcor_set_dict)
        end

        # compute correlations on the current subset of variables
        if !isempty(test_obj.cor_mat)
            cor_subset!(data, test_obj.cor_mat, [X, Y, Z_total...])

            debug > 2 && println(test_obj.cor_mat[[X, Y, Z_total...], [X, Y, Z_total...]])
        end
    end

    num_tests_total = 0
    for subset_size in max_k:-1:1
        Z_combos = combinations(Z_total, subset_size)
        num_tests_total += length(Z_combos)

        for Zs_arr in Z_combos
            Zs = Tuple(Zs_arr)
            if discrete_test
                test_result = test(X, Y, Zs, data, test_obj, hps, z)
            else
                test_result = test(X, Y, Zs, data, test_obj, n_obs_min)
            end
            num_tests += 1

            debug > 2 && println("\t subset ", Zs, " : ", test_result)

            if !issig(test_result, alpha; test_obj=test_obj) || (max_tests > 0 && num_tests >= max_tests)
                if subset_size > 1
                    for remaining_subset_size in subset_size-1:-1:1
                        num_tests_total += length(combinations(Z_total, remaining_subset_size))
                    end
                end
                test_fraction = num_tests / num_tests_total

                max_tests > 0 && num_tests >= max_tests && @warn "Maximum number of tests for variable pair $X / $Y at $num_tests out of $num_tests_total tests (fraction: $(round(test_fraction, digits=3)), size of Z: $(length(Z_total)))."

                return test_result, Zs, num_tests, test_fraction

            elseif test_result.pval >= lowest_sig_result.pval
                lowest_sig_result = test_result
                lowest_sig_Zs = Zs
            end
        end
    end

    lowest_sig_result, lowest_sig_Zs, num_tests, num_tests / num_tests_total
end


function test_subsets(itr::BNBIterator, alpha::AbstractFloat; n_obs_min::Integer=0, max_tests::Integer=1e9, test_obj::AbstractTest)
    num_tests = 0
    if n_obs_min > size(itr.data, 1)
        return TestResult(0.0, 1.0, 0.0, false), (), num_tests
    else
        ret_test_res = TestResult(1.0, 0.0, 0.0, true)
        ret_Zs = ()
        for (test_res, Zs) in itr
            num_tests += 1
            if !issig(test_res, alpha; test_obj=test_obj) || (max_tests > 0 && num_tests >= max_tests)
                return test_res, Zs, num_tests
            elseif test_res.pval > ret_test_res.pval
                ret_test_res = test_res
                ret_Zs = Zs
            end
        end
        return ret_test_res, ret_Zs, num_tests
    end
end


# backend functions for pairwise univariate tests

function condensed_stats_to_dict(n_vars::Integer, pvals::AbstractVector{Float64}, stats::AbstractVector{Float64},
     alpha::AbstractFloat)

    nbr_dict = Dict([(X, OrderedDict{Int,Tuple{Float64,Float64}}()) for X in 1:n_vars])

    for X in 1:n_vars-1, Y in X+1:n_vars
        pair_index = sum(n_vars-1:-1:n_vars-X) - n_vars + Y
        pval = pvals[pair_index]

        if !isnan(pval) && pval < alpha
            stat = stats[pair_index]
            nbr_dict[X][Y] = (stat, pval)
            nbr_dict[Y][X] = (stat, pval)
        end
    end
    nbr_dict
end


function add_pwresults_to_matrix!(X, Ys, test_results, stats, pvals, n_vars,
    correct_reliable_only)

    for (Y, test_res) in zip(Ys, test_results)
        pair_index = sum(n_vars-1:-1:n_vars-X) - n_vars + Y

        if correct_reliable_only && !test_res.suff_power
            curr_stat = curr_pval = NaN64
        else
            curr_stat = test_res.stat
            curr_pval = test_res.pval
        end

        stats[pair_index] = curr_stat
        pvals[pair_index] = curr_pval
    end
end


function pw_univar_kernel(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{ElType},
    test_obj::AbstractTest, hps::Integer, n_obs_min::Integer) where ElType <: Real
    if needs_nz_view(X, data, test_obj)
        sub_data = @view data[data[:, X] .!= 0, :]
    else
        sub_data = data
    end

    if isdiscrete(test_obj)
        test_results = test(X, Ys, sub_data, test_obj, hps, n_obs_min)
    else
        test_results = test(X, Ys, sub_data, test_obj, n_obs_min)
    end
end


function pw_univar_kernel!(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{ElType},
            stats::AbstractVector{Float64}, pvals::AbstractVector{Float64},
            test_obj::AbstractTest, hps::Integer, n_obs_min::Integer,
            correct_reliable_only::Bool=false) where ElType <: Real
    test_results = pw_univar_kernel(X, Ys, data, test_obj, hps, n_obs_min)
    add_pwresults_to_matrix!(X, Ys, test_results, stats, pvals, size(data, 2),
                             correct_reliable_only)
end


function pw_univar_neighbors(data::AbstractMatrix{ElType};
        test_name::String="mi", alpha::Float64=0.01, hps::Int=5, n_obs_min::Int=0, FDR::Bool=true,
        levels::AbstractVector{DiscType}=DiscType[], max_vals::AbstractVector{DiscType}=DiscType[],
        parallel::String="single",
        workers_local::Bool=true,
        cor_mat::Matrix{ContType}=zeros(ContType, 0, 0),
        tmp_folder::AbstractString="",
        correct_reliable_only::Bool=true, shuffle_jobs::Bool=true,
        pmap_batch_size=nothing) where {ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}

    target_vars = collect(1:size(data, 2))

    if startswith(test_name, "mi") && isempty(levels)
        levels = get_levels(data)
        max_vals = get_max_vals(data)
    end

    test_obj = make_test_object(test_name, false, levels=levels, max_vals=max_vals, cor_mat=cor_mat)

    n_vars = length(target_vars)
    n_pairs = convert(Int, n_vars * (n_vars - 1) / 2)

    if isnothing(pmap_batch_size)
        pmap_batch_size = Int(ceil(n_vars / (nprocs() * 3)))
    end

    work_items = [(X, X+1:n_vars) for X in 1:n_vars-1]
    pvals = fill(NaN64, n_pairs)
    stats = fill(NaN64, n_pairs)

    # no need to start workers and send data if the correlation matrix
    # was already precomputed
    if test_name == "fz" && !isempty(cor_mat)
        parallel = "single_il"
    end

    if startswith(parallel, "single")
        for (X, Ys_slice) in work_items
            pw_univar_kernel!(X, Ys_slice, data, stats, pvals, test_obj, hps, n_obs_min,
                              correct_reliable_only)
        end

    else
        if shuffle_jobs
            shuffle!(work_items)
        end

        # if worker processes are on the same machine, use local memory sharing via shared arrays
        if workers_local
            shared_pvals = SharedArray{Float64}(pvals)
            shared_stats = SharedArray{Float64}(stats)

            wp = CachingPool(workers())
            let data=data, test_obj=test_obj
                pmap(work_item -> pw_univar_kernel!(work_item..., data, shared_stats,
                                                    shared_pvals, test_obj, hps,
                                                    n_obs_min, correct_reliable_only),
                                                    wp, work_items,
                                                    batch_size=pmap_batch_size)
            end

            stats = shared_stats.s
            pvals = shared_pvals.s

        # otherwise make workers store test results remotely and gather them
        # in the end via network
        else
            wp = CachingPool(workers())
            test_result_chunks = let data=data, test_obj=test_obj
                pmap(work_item -> pw_univar_kernel(work_item..., data,
                                                   test_obj, hps, n_obs_min),
                                                   work_items,
                                                   batch_size=pmap_batch_size)
            end

            for ((X, Ys), test_results) in zip(work_items, test_result_chunks)
                add_pwresults_to_matrix!(X, Ys, test_results, stats, pvals, n_vars,
                                         correct_reliable_only)
            end
        end
    end

    if FDR
        m = length(pvals)

        if correct_reliable_only
            m -= sum(isnan.(pvals))
        end

        benjamini_hochberg!(pvals, alpha=alpha, m=m)
    end

    condensed_stats_to_dict(n_vars, pvals, stats, alpha)
end


function _trim_mutual_kernel(X, Y, Zs_total, data, test_obj, alpha, hps, z, n_obs_min)
    if needs_nz_view(X, data, test_obj)
        data = prepare_nzdata(X, data, test_obj)
    end

    if needs_nz_view(Y, data, test_obj)
        data = prepare_nzdata(Y, data, test_obj)
    end

    if iscontinuous(test_obj) && is_zero_adjusted(test_obj) && !isempty(test_obj.cor_mat)
        cor_subset!(data, test_obj.cor_mat, [X, Y, Zs_total...])
    end

    all_test_res = Vector{TestResult}(undef, length(Zs_total))
    for (i, Z) in enumerate(Zs_total)
        test_res = if isdiscrete(test_obj)
            test(X, Y, (Z,), data, test_obj, hps, z)
        else
            test(X, Y, (Z,), data, test_obj, n_obs_min)
        end
        all_test_res[i] = test_res
    end
    
    return filter(((Z, test_res),)->!issig(test_res, alpha; test_obj=test_obj), collect(zip(Zs_total, all_test_res)))
end

function trim_mutual_discards!(nbr_dict::Dict{Int,HitonState{Int}}, all_univar_nbrs::Dict{Int,NbrStatDict}, data::AbstractMatrix, test_name::String; parallel::String="single_il", 
    alpha::AbstractFloat, hps::Integer=5, n_obs_min::Integer=0, z::Vector{<:Integer}=Int[], 
    cache_pcor::Bool=true, levels, max_vals, cor_mat, track_rejections)

    test_pair_dict = Dict{Tuple{Int,Int},Set{Int}}() 
    for (X, nbr_state) in nbr_dict
        Ys_cond = Set(keys(nbr_state.state_results))
        Ys_uni = setdiff(Set(keys(all_univar_nbrs[X])), Ys_cond)

        for Y_cond in Ys_cond
            test_pair = X < Y_cond ? (X, Y_cond) : (Y_cond, X)
            if !haskey(test_pair_dict, test_pair)
                test_pair_dict[test_pair] = Set{Int}()
            end
            test_candidates = test_pair_dict[test_pair]

            for Y_uni in Ys_uni
                push!(test_candidates, Y_uni)
            end
        end
    end
    
    test_obj = make_test_object(test_name, true, max_k=1, levels=levels, max_vals=max_vals, cor_mat=cor_mat, cache_pcor=cache_pcor)
    work_units = [(XY, collect(Zs_total)) for (XY, Zs_total) in test_pair_dict]
    tests_nonsig = if startswith(parallel, "single")
        map(((XY, Zs_total),)->_trim_mutual_kernel(XY..., Zs_total, data, test_obj, alpha, hps, z, n_obs_min), work_units)
    else
        wp = CachingPool(workers())
        pmap(((XY, Zs_total),)->_trim_mutual_kernel(XY..., Zs_total, data, test_obj, alpha, hps, z, n_obs_min), wp, work_units)
    end

    #@show tests_nonsig first.(work_units)

    for ((X, Y), curr_tests_nonsig) in zip(first.(work_units), tests_nonsig)
        for (Z_nonsig, test_res_nonsig) in curr_tests_nonsig
            for (T1, T2) in [(X, Y), (Y, X)]
                !haskey(nbr_dict, T1) && continue
                nbr_state = nbr_dict[T1]
                if haskey(nbr_state.state_results, T2)
                    # remove partner
                    delete!(nbr_state.state_results, T2)
                    
                    # add entry with rejection information for partner
                    if track_rejections
                        nbr_state.state_rejections[T2] = ((Z_nonsig,), test_res_nonsig, (0, 0.0))
                    end

                    # remove partner from unchecked variables
                    ind = findfirst(==(T2), nbr_state.unchecked_vars)
                    if !isnothing(ind)
                        deleteat!(nbr_state.unchecked_vars, ind)
                    end
                end
            end
        end
    end
end