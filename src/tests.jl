module Tests

export test, test_subsets, issig, pw_univar_neighbors

using Combinatorics
using DataStructures

using FlashWeave.Types
using FlashWeave.Misc
using FlashWeave.Statfuns
using FlashWeave.Contingency


function issig(test_res::TestResult, alpha::AbstractFloat)
    test_res.pval < alpha && test_res.suff_power == true
end


sufficient_power(levels_x::Integer, levels_y::Integer, n_obs::Integer, hps::Integer) = (n_obs / (levels_x * levels_y)) > hps
sufficient_power(levels_x::Integer, levels_y::Integer, levels_z::Integer, n_obs::Integer, hps::Integer) = (n_obs / (levels_x * levels_y * levels_z)) > hps

##################
### UNIVARIATE ###
##################

### discrete

function test(X::Int, Y::Int, data::AbstractMatrix{<:Integer}, test_obj::AbstractContTest, hps::Integer,
        n_obs_min::Int=0)
    @inbounds levels_x = test_obj.levels[X]
    @inbounds levels_y = test_obj.levels[Y]

    if !issparse(data)
        contingency_table!(X, Y, data, test_obj.ctab)
    else
        contingency_table!(X, Y, data, test_obj)
    end

    if is_zero_adjusted(test_obj)
        sub_ctab = nz_adjust_cont_tab(levels_x, levels_y, test_obj.ctab)
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
        pval = mi_pval(mi_stat, df, n_obs)
        suff_power = true

        # use oddsratio of 2x2 contingency table to determine edge sign
        mi_sign = oddsratio(sub_ctab) < 1.0 ? -1.0 : 1.0
        mi_stat *= mi_sign
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
        return [TestResult(0.0, 1.0, 0, false) for Y in Ys]
    else
        return map(Y -> test(X, Y, data, test_obj, hps, n_obs_min), Ys)
    end
end

# convenience wrapper
function test(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{<:Integer},
        test_name::String, hps::Integer=5, n_obs_min::Int=0, levels::Vector{<:Integer}=Int[])
    if isempty(levels)
        levels = get_levels(data)#map(x -> length(unique(data[:, x])), 1:size(data, 2))
    end
    test_obj = make_test_object(test_name, false, max_k=0, levels=levels, cor_mat=zeros(Float64, 0, 0))
    test(X, Ys, data, test_obj, hps, n_obs_min)
end


### continuous

function test(X::Int, Y::Int, data::AbstractMatrix{<:Real}, test_obj::FzTest,
        n_obs_min::Integer=0, Y_adjusted::Bool=false)

    if isempty(data)
        p_stat = 0.0
        df = 0
        pval = 1.0
        n_obs = 0
    else
        nz = is_zero_adjusted(test_obj)

        @inbounds if isempty(test_obj.cor_mat)
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

    map(Y -> test(X, Y, data, test_obj, n_obs_min, false), Ys)
end

#convenience wrapper
function test(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{<:AbstractFloat},
        test_name::String, n_obs_min::Integer=0)
    test_obj = make_test_object(test_name, false, max_k=0, levels=Int[], cor_mat=zeros(Float64, 0, 0))
    test(X, Ys, data, test_obj, n_obs_min)
end

###################
### CONDITIONAL ###
###################

function test(X::Int, Y::Int, Zs::Tuple{Vararg{Int64,N} where N<:Int}, data::AbstractMatrix{<:Integer}, test_obj::MiTestCond, hps::Integer, z::AbstractVector{<:Integer}=Int[])
    """Test association between X and Y"""
    @inbounds levels_x = test_obj.levels[X]
    @inbounds levels_y = test_obj.levels[Y]

    if !issparse(data)
        levels_z = contingency_table!(X, Y, Zs, data, test_obj.ctab, z, test_obj.zmap.cum_levels, test_obj.zmap.z_map_arr)
    else
        contingency_table!(X, Y, Zs, data, test_obj)
        levels_z = test_obj.zmap.levels_total
    end

    if is_zero_adjusted(test_obj)
        sub_ctab = nz_adjust_cont_tab(levels_x, levels_y, test_obj.ctab)
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
        pval = mi_pval(mi_stat, df, n_obs)
        suff_power = true

        # use oddsratio of 2x2 contingency table to determine edge sign
        mi_sign = oddsratio(sub_ctab) < 1.0 ? -1.0 : 1.0
        mi_stat *= mi_sign
    end

    Misc.TestResult(mi_stat, pval, df, suff_power)
end


function test(X::Int, Y::Int, Zs::Tuple{Vararg{Int64,N} where N<:Int}, data::AbstractMatrix{<:Integer}, test_name::String, hps::Integer=5)
    """Convenience function for module tests"""
    levels = get_levels(data)
    test_obj = MiTestCond(levels, is_zero_adjusted(test_name) ? Nz() : NoNz(), length(Zs))

    z = issparse(data) ? eltype(levels)[] : zeros(eltype(levels), size(data, 1))
    test(X, Y, Zs, data, test_obj, hps, z)
end



## CONTINUOUS ##

function test(X::Int, Y::Int, Zs::Tuple{Vararg{Int64,N} where N<:Int}, data::AbstractMatrix{<:AbstractFloat},
    test_obj::FzTestCond, n_obs_min::Integer)
    """Critical: expects zeros to be trimmed from both X and Y if nz is true"""

    n_obs = size(data, 1)

    if n_obs >= n_obs_min
        p_stat = isempty(test_obj.cor_mat) ? pcor(X, Y, Zs, data) : pcor_rec(X, Y, Zs, test_obj.cor_mat, test_obj.pcor_set_dict,
            test_obj.cache_pcor)
        pval = fz_pval(p_stat, n_obs, 0)
    else
        p_stat = 0.0
        pval = 1.0
    end

    df = 0

    Misc.TestResult(p_stat, pval, df, n_obs >= n_obs_min)
end

# convenience function for module tests

function test(X::Int, Y::Int, Zs::Tuple{Vararg{Int64,N} where N<:Int}, data::AbstractMatrix{<:AbstractFloat}, test_name::String;
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

    lowest_sig_result = TestResult(0.0, 0.0, 0.0, true)
    lowest_sig_Zs = ()
    discrete_test = isdiscrete(test_obj)
    num_tests = 0
    nz = is_zero_adjusted(test_obj)

    if !discrete_test && nz
        if n_obs_min > size(data, 1)
            return TestResult(0.0, 1.0, 0.0, false), Int[]
        end

        if test_obj.cache_pcor
            empty!(test_obj.pcor_set_dict)
        end

        # compute correlations on the current subset of variables
        if !isempty(test_obj.cor_mat)
            cor_subset!(data, test_obj.cor_mat, [X, Y, Z_total...])

            if debug > 2
                println(test_obj.cor_mat[[X, Y, Z_total...], [X, Y, Z_total...]])
            end
        end
    end

    num_tests_total = 0
    for subset_size in max_k:-1:1
        Z_combos = combinations(Z_total, subset_size)#isempty(Z_wanted) ? combinations(Z_total, subset_size) : combinations_with_whitelist(Z_total, Z_wanted, subset_size)
        num_tests_total += length(Z_combos)

        for Zs_arr in Z_combos
            Zs = Tuple(Zs_arr)
            if discrete_test
                test_result = test(X, Y, Zs, data, test_obj, hps, z)
            else
                test_result = test(X, Y, Zs, data, test_obj, n_obs_min)
            end
            num_tests += 1

            if debug > 2
                println("\t subset ", Zs, " : ", test_result)
            end

            if !issig(test_result, alpha) || num_tests >= max_tests
                if subset_size > 1
                    for remaining_subset_size in subset_size-1:-1:1
                        num_tests_total += length(combinations(Z_total, remaining_subset_size))
                    end
                end
                test_fraction = num_tests / num_tests_total

                num_tests >= max_tests && warn("Maximum number of tests for variable pair $X / $Y at $num_tests out fo $num_tests_total tests (fraction: $(round(test_fraction, 3))).")

                return test_result, Zs, num_tests, test_fraction

            elseif test_result.pval >= lowest_sig_result.pval
                lowest_sig_result = test_result
                lowest_sig_Zs = Zs
            end
        end
    end

    lowest_sig_result, lowest_sig_Zs, num_tests, num_tests / num_tests_total
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


function pw_univar_kernel!{ElType <: Real}(X::Int, Ys_slice::AbstractVector{Int}, data::AbstractMatrix{ElType},
                            stats::AbstractVector{Float64}, pvals::AbstractVector{Float64},
                            test_obj::AbstractTest, hps::Integer, n_obs_min::Integer,
                            correct_reliable_only::Bool=false)
    n_vars = size(data, 2)

    if needs_nz_view(X, data, test_obj)
        sub_data = @view data[data[:, X] .!= 0, :]
    else
        sub_data = data
    end

    Ys = collect(Ys_slice)

    if isdiscrete(test_obj)
        test_results = test(X, Ys, sub_data, test_obj, hps, n_obs_min)
    else
        test_results = test(X, Ys, sub_data, test_obj, n_obs_min)
    end

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

function pw_univar_kernel{ElType <: Real}(X::Int, Ys_slice::AbstractVector{Int}, data::AbstractMatrix{ElType},
                            test_obj::AbstractTest, hps::Integer, n_obs_min::Integer)
    n_vars = size(data, 2)

    if needs_nz_view(X, data, test_obj)
        sub_data = @view data[data[:, X] .!= 0, :]
    else
        sub_data = data
    end

    Ys = collect(Ys_slice)

    if isdiscrete(test_obj)
        test_results = test(X, Ys, sub_data, test_obj, hps, n_obs_min)
    else
        test_results = test(X, Ys, sub_data, test_obj, n_obs_min)
    end
end


function pw_univar_neighbors{ElType<:Real, DiscType<:Integer, ContType<:AbstractFloat}(data::AbstractMatrix{ElType};
        test_name::String="mi", alpha::Float64=0.01, hps::Int=5, n_obs_min::Int=0, FDR::Bool=true,
        levels::AbstractVector{DiscType}=DiscType[], parallel::String="single", workers_local::Bool=true,
        cor_mat::Matrix{ContType}=zeros(ContType, 0, 0),
        chunk_size::Int=500, correct_reliable_only::Bool=true)


    target_vars = collect(1:size(data, 2))

    if startswith(test_name, "mi") && isempty(levels)
        levels = map(x -> get_levels(data[:, x]), target_vars)
    end

    test_obj = make_test_object(test_name, false, levels=levels, cor_mat=cor_mat)

    n_vars = length(target_vars)
    n_pairs = convert(Int, n_vars * (n_vars - 1) / 2)

    nz = is_zero_adjusted(test_obj)


    effective_chunk_size = min(chunk_size, Int(ceil(n_vars / 3)))
    work_items = collect(work_chunker(n_vars, effective_chunk_size))

    pvals = fill(NaN64, n_pairs)#ones(Float64, n_pairs)
    stats = fill(NaN64, n_pairs)#zeros(Float64, n_pairs)

    if startswith(parallel, "single")
        for (X, Ys_slice) in work_items
            pw_univar_kernel!(X, Ys_slice, data, stats, pvals, test_obj, hps, n_obs_min, correct_reliable_only)
        end

    else
        shuffle!(work_items)
        if startswith(parallel, "multi")
            # if worker processes are on the same machine, use local memory sharing via shared arrays
            if workers_local
                shared_pvals = SharedArray{Float64}(pvals)
                shared_stats = SharedArray{Float64}(stats)
                @sync @parallel for work_item in work_items
                    pw_univar_kernel!(work_item[1], work_item[2], data, shared_stats, shared_pvals, test_obj, hps,
                                      n_obs_min, correct_reliable_only)
                end
                stats = shared_stats.s
                pvals = shared_pvals.s

            # otherwise make workers store test results remotely and gather them in the end via network
            else
                all_test_results = @parallel (vcat) for work_item in work_items
                    pw_univar_kernel(work_item[1], work_item[2], data, test_obj, hps, n_obs_min)
                end

                i = 0
                for (X, Ys_slice) in work_items
                    for Y in Ys_slice
                        i += 1
                        test_res = all_test_results[i]
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
            end

        elseif startswith(parallel, "threads")
            pvals = fill(NaN64, n_pairs)#ones(Float64, n_pairs)
            stats = fill(NaN64, n_pairs)#zeros(Float64, n_pairs)
            Threads.@threads for work_item in work_items
                pw_univar_kernel!(work_item[1], work_item[2], data, stats, pvals, test_obj, hps, n_obs_min,
                                  correct_reliable_only)
            end
        end
    end

    if FDR
        if correct_reliable_only && any(isnan(x) for x in pvals)
            reliable_mask = .!isnan.(pvals)
            reliable_pvals = pvals[reliable_mask]
            reliable_pvals = benjamini_hochberg(reliable_pvals)

            rel_pval_i = 1
            for (i, is_reliable_elem) in enumerate(reliable_mask)
                if is_reliable_elem
                    pvals[i] = reliable_pvals[rel_pval_i]
                    rel_pval_i += 1
                else
                    pvals[i] = NaN64
                end
            end
        else
            pvals = benjamini_hochberg(pvals)
        end
    end

    condensed_stats_to_dict(n_vars, pvals, stats, alpha)
end


end
