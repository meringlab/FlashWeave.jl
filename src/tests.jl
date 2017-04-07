module Tests

export test, test_subsets, issig

using Combinatorics

using Cauocc.Misc
using Cauocc.Statfuns
using Cauocc.Contingency


function issig(test_res::TestResult, alpha::Float64)
    test_res.pval < alpha && test_res.suff_power == true
end


sufficient_power(levels_x::Int, levels_y::Int, n_obs::Int, hps::Int) = (n_obs / (levels_x * levels_y)) > hps
sufficient_power(levels_x::Int, levels_y::Int, levels_z::Int, n_obs::Int, hps::Int) = (n_obs / (levels_x * levels_y * levels_z)) > hps


function test(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Float64},SparseMatrixCSC{Float64,Int64}},
        test_name::String)
    
    if test_name == "fz" || test_name == "fz_nz"
        p_stat = pcor(X, Y, Zs, data)
        df = 0
        pval = fz_pval(p_stat, size(data, 1), 0)
    end
    Misc.TestResult(p_stat, pval, df, true)
end


function test(X::Int, Y::Int, Zs::Vector{Int}, data::Union{SubArray,Matrix{Int64},SparseMatrixCSC{Int64,Int64}},
        test_name::String, hps::Int, levels_x::Int, levels_y::Int, cont_tab::Array{Int,3},
    z::Vector{Int}, ni::Array{Int,2}, nj::Array{Int,2}, nk::Array{Int,1}, cum_levels::Vector{Int},
    z_map_arr::Vector{Int}, nz::Bool=false, data_row_inds::Vector{Int64}=Int64[], data_nzero_vals::Vector{Int64}=Int64[],
    levels::Vector{Int64}=Int64[])
    """Test association between X and Y"""
    
    if !issparse(data)
        levels_z = contingency_table!(X, Y, Zs, data, cont_tab, z, cum_levels, z_map_arr)
    else
        Zs_tup = tuple(Zs...)
        cont_levels = nz ? levels : nothing
        levels_z = contingency_table!(X, Y, Zs_tup, data, data_row_inds, data_nzero_vals, cont_tab, cum_levels, z_map_arr,
                                      cont_levels)
    end
    
    if nz
        sub_cont_tab = nz_adjust_cont_tab(levels_x, levels_y, cont_tab)
        levels_x = size(sub_cont_tab, 1)
        levels_y = size(sub_cont_tab, 2)
    else
        sub_cont_tab = cont_tab
    end
    
    n_obs = sum(sub_cont_tab)
    
    if is_mi_test(test_name)
        if !sufficient_power(levels_x, levels_y, levels_z, n_obs, hps)
            mi_stat = 0.0
            df = 0
            pval = 1.0
            suff_power = false
        else
            mi_stat = mutual_information(sub_cont_tab, levels_x, levels_y, levels_z, ni, nj, nk, test_name == "mi_expdz")
            
            df = adjust_df(ni, nj, levels_x, levels_y, levels_z)
            pval = mi_pval(mi_stat, df, n_obs)
            suff_power = true
            
            # use oddsratio of 2x2 contingency table to determine edge sign
            mi_sign = oddsratio(sub_cont_tab) < 1.0 ? -1.0 : 1.0
            mi_stat *= mi_sign
        end
    end
    Misc.TestResult(mi_stat, pval, df, suff_power)
end


function test(X::Int, Y::Int, data::Union{SubArray,Matrix{Int64},SparseMatrixCSC{Int64,Int64}}, test_name::String, hps::Int,
    levels_x::Int, levels_y::Int, cont_tab::Array{Int,2}, ni::Array{Int,1}, nj::Array{Int,1}, nz::Bool=false,
    data_row_inds::Vector{Int64}=Int64[], data_nzero_vals::Vector{Int64}=Int64[])
        
    if nz && !issparse(data) && (levels_y > 2)
        sub_data = @view data[data[:, Y] .!= 0, :]
    else
        sub_data = data
    end
    
    if !issparse(data)
        contingency_table!(X, Y, sub_data, cont_tab)
    else
        contingency_table!(X, Y, sub_data, data_row_inds, data_nzero_vals, cont_tab)
    end
    
    if nz
        sub_cont_tab = nz_adjust_cont_tab(levels_x, levels_y, cont_tab)
        levels_x = size(sub_cont_tab, 1)
        levels_y = size(sub_cont_tab, 2)
    else
        sub_cont_tab = cont_tab
    end
    
    n_obs = sum(sub_cont_tab)
    
    if is_mi_test(test_name)
        if !sufficient_power(levels_x, levels_y, n_obs, hps)
            mi_stat = 0.0
            df = 0
            pval = 1.0
            suff_power = false
        else
            mi_stat = mutual_information(sub_cont_tab, levels_x, levels_y, ni, nj, test_name == "mi_expdz")
            
            df = adjust_df(ni, nj, levels_x, levels_y)
            pval = mi_pval(mi_stat, df, n_obs)
            suff_power = true
            
            # use oddsratio of 2x2 contingency table to determine edge sign
            mi_sign = oddsratio(sub_cont_tab) < 1.0 ? -1.0 : 1.0
            mi_stat *= mi_sign
        end
    end
    TestResult(mi_stat, pval, df, suff_power) 
end


function test(X::Int, Y::Int, data::Union{SubArray,Matrix{Float64},SparseMatrixCSC{Float64,Int64}}, test_name::String)
    
    if test_name == "fz" || test_name == "fz_nz"
        p_stat = cor(data[:, X], data[:, Y])
        df = 0
        pval = fz_pval(p_stat, size(data, 1), 0)
    end
    TestResult(p_stat, pval, df, true)
end


function test(X::Int, Ys::Vector{Int}, data::Union{SubArray,Matrix{Int64},SparseMatrixCSC{Int64,Int64}}, test_name::String,
    hps::Int, levels::Vector{Int}, data_row_inds::Vector{Int64}=Int64[], data_nzero_vals::Vector{Int64}=Int64[])
    """Test all variables Ys for univariate association with X"""
    
    levels_x = levels[X]
    
    if levels_x < 2
        return [TestResult(0.0, 1.0, 0, false) for Y in Ys]
    else
        max_level_y = maximum(levels[Ys])
        cont_tab = zeros(Int, levels_x, max_level_y)
        ni = zeros(Int, max_level_y)
        nj = zeros(Int, max_level_y)
        nz = is_zero_adjusted(test_name)

        return map(Y -> test(X, Y, data, test_name, hps, levels_x, levels[Y], cont_tab, ni, nj, nz, data_row_inds, data_nzero_vals), Ys)
    end
end


function test(X::Int, Ys::Array{Int, 1}, data::Union{SubArray,Matrix{Float64},SparseMatrixCSC{Float64,Int64}}, test_name::String)
    """Test all variables Ys for univariate association with X"""
    map(Y -> test(X, Y, data, test_name), Ys)
end


function test_subsets(X::Int, Y::Int, Z_total::Vector{Int}, data,
    test_name::String, max_k::Int, alpha::Float64; hps::Int=5, pwr::Float64=0.5, levels::Vector{Int}=Int[],
    data_row_inds::Vector{Int64}=Int64[], data_nzero_vals::Vector{Int64}=Int64[])
    lowest_sig_result = TestResult(0.0, 0.0, 0.0, true)
    discrete_test = isdiscrete(test_name)
    num_tests = 0
    
    if discrete_test       
        levels_x = levels[X]
        levels_y = levels[Y]
        max_levels = maximum(levels)
        max_levels_z = sum([max_levels^(i+1) for i in 1:max_k])
        cont_tab = zeros(Int, levels_x, levels_y, max_levels_z)
        z = zeros(Int, size(data, 1))
        ni = zeros(Int, levels_x, max_levels_z)
        nj = zeros(Int, levels_y, max_levels_z)
        nk = zeros(Int, max_levels_z)
        cum_levels = zeros(Int, max_k + 1)
        z_map_arr = zeros(Int, max_levels_z)
        num_lowpwr_tests = 0
        nz = is_zero_adjusted(test_name)
    end
    
    for subset_size in 1:max_k
        Z_combos = combinations(Z_total, subset_size)
        
        for Zs in Z_combos
            if discrete_test
                make_cum_levels!(cum_levels, Zs, levels)
                test_result = test(X, Y, Zs, data, test_name, hps, levels_x, levels_y, cont_tab, z,
                                   ni, nj, nk, cum_levels, z_map_arr, nz, data_row_inds, data_nzero_vals, levels)
            else
                test_result = test(X, Y, Zs, data, test_name)
            end
            num_tests += 1
            
            # if discrete test didn't have enough power, check if
            # the threshold of number of unreliable tests has been reached
            if discrete_test & !test_result.suff_power
                num_lowpwr_tests += 1
                
                if num_lowpwr_tests / num_tests >= 1 - pwr
                    lowest_sig_result.suff_power = false
                    return lowest_sig_result
                end
            else
                if !issig(test_result, alpha)
                    #print("[test_subsets] rejected $X with $Y with rejecting subset $Zs")
                    return test_result
                elseif test_result.pval > lowest_sig_result.pval
                    lowest_sig_result = test_result
                end
            end
        end
    end
    
    lowest_sig_result
end

end