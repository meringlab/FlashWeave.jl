module Tests

export test, test_subsets, issig

using Combinatorics

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
    levels_x = test_obj.levels[X]
    levels_y = test_obj.levels[Y]
    
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

    if test_obj.levels[X] < 2
        return [TestResult(0.0, 1.0, 0, false) for Y in Ys]
    else
        return map(Y -> test(X, Y, data, test_obj, hps, n_obs_min), Ys)     
    end
end

# convenience wrapper
function test(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{<:Integer},
        test_name::String, hps::Integer=5, n_obs_min::Int=0)
    levels = map(x -> length(unique(data[:, x])), 1:size(data, 2))
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
        
        if isempty(test_obj.cor_mat)
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


function test(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{<:Real},
        test_obj::AbstractCorTest, n_obs_min::Integer=0)
    """CRITICAL: expects zeros to be trimmed from X if nz_test
    is provided!
    
    Test all variables Ys for univariate association with X"""
    
    map(Y -> test(X, Y, data, test_obj, n_obs_min, false), Ys)
end

#convenience wrapper
function test(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{<:Real},
        test_name::String, n_obs_min::Integer=0)
    test_obj = make_test_object(test_name, false, max_k=0, levels=Int[], cor_mat=zeros(Float64, 0, 0))
    test(X, Ys, data, test_obj, n_obs_min)
end

###################
### CONDITIONAL ###
###################

function test(X::Int, Y::Int, Zs::AbstractVector{Int}, data::AbstractMatrix{<:Integer}, test_obj::MiTestCond, hps::Integer, z::AbstractVector{<:Integer}=Int[])
    """Test association between X and Y"""
    levels_x = test_obj.levels[X]
    levels_y = test_obj.levels[Y]
    
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


function test(X::Int, Y::Int, Zs::AbstractVector{Int}, data::AbstractMatrix{<:Integer}, test_name::String, hps::Integer=5)
    levels = get_levels(data)
    test_obj = MiTestCond(levels, is_zero_adjusted(test_name) ? Nz() : NoNz(), length(Zs))

    z = issparse(data) ? eltype(levels)[] : zeros(eltype(levels), size(data, 1))
    test(X, Y, Zs, data, test_obj, hps, z)
end



## CONTINUOUS ##

function test(X::Int, Y::Int, Zs::AbstractVector{Int}, data::AbstractMatrix{<:AbstractFloat},
    test_obj::FzTestCond, n_obs_min::Integer, cache_result::Bool=true)
    """Critical: expects zeros to be trimmed from both X and Y if nz is true"""

    n_obs = size(data, 1)

    if n_obs >= n_obs_min
        p_stat = isempty(test_obj.cor_mat) ? pcor(X, Y, Zs, data) : pcor_rec(X, Y, Zs, test_obj.cor_mat, test_obj.pcor_set_dict, cache_result)
        pval = fz_pval(p_stat, n_obs, 0)
    else
        p_stat = 0.0
        pval = 1.0
    end

    df = 0
    
    Misc.TestResult(p_stat, pval, df, n_obs >= n_obs_min)
end


function test(X::Int, Y::Int, Zs::AbstractVector{Int}, data::AbstractMatrix{<:Real}, test_name::String; recursive::Bool=true, n_obs_min::Integer=0)
    cor_mat = recursive ? cor(data) : zeros(Float64, 0, 0)
    test_obj = FzTestCond(cor_mat, Dict{String,Dict{String,eltype(cor_mat)}}(), is_zero_adjusted(test_name) ? Nz() : NoNz())
    test(X, Y, Zs, data, test_obj, n_obs_min, true)
end


## MAIN SUBSET TEST FUNCTION ##

function test_subsets(X::Int, Y::Int, Z_total::AbstractVector{Int}, data::AbstractMatrix{<:Real},
    test_obj::AbstractTest, max_k::Integer, alpha::AbstractFloat; hps::Integer=5, n_obs_min::Integer=0,
    debug::Int=0, Z_wanted::AbstractVector{Int}=Int[], z::Vector{<:Integer}=Int[])

    lowest_sig_result = TestResult(0.0, 0.0, 0.0, true)
    lowest_sig_Zs = Int[]
    discrete_test = isdiscrete(test_obj)
    num_tests = 0
    nz = is_zero_adjusted(test_obj)

    if !discrete_test && nz
        if n_obs_min > size(data, 1)
            return TestResult(0.0, 1.0, 0.0, false), Int[]
        end      
            
        empty!(test_obj.pcor_set_dict)

        # compute correlations on the current subset of variables
        if !isempty(test_obj.cor_mat)
            cor_subset!(data, test_obj.cor_mat, [X, Y, Z_total...])
            
            if debug > 2
                println(test_obj.cor_mat[[X, Y, Z_total...], [X, Y, Z_total...]])
            end
        end
    end
    
    for subset_size in max_k:-1:1
        Z_combos = isempty(Z_wanted) ? combinations(Z_total, subset_size) : combinations_with_whitelist(Z_total, Z_wanted, subset_size)
        
        for Zs in Z_combos
            if discrete_test
                test_result = test(X, Y, Zs, data, test_obj, hps, z)
            else
                test_result = test(X, Y, Zs, data, test_obj, n_obs_min, subset_size < max_k)
            end
            num_tests += 1

            if debug > 2
                println("\t subset ", Zs, " : ", test_result)
            end
            
            if !issig(test_result, alpha)
                return test_result, Zs
            elseif test_result.pval >= lowest_sig_result.pval
                lowest_sig_result = test_result
                lowest_sig_Zs = Zs
            end
        end
    end

    lowest_sig_result, lowest_sig_Zs
end


end
