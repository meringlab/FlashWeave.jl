module Tests

export test, test_subsets, issig

using Combinatorics

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

function test{ElType <: Integer}(X::Int, Y::Int, data::AbstractMatrix{ElType}, test_name::String, hps::Integer,
    levels_x::ElType, levels_y::ElType, cont_tab::Matrix{ElType}, ni::AbstractVector{ElType}, nj::AbstractVector{ElType}, nz::Bool=false)

    #if needs_nz_view(candidate, data, nz, levels)
    #    sub_data = @view data[data[:, Y] .!= 0, :]
    #else
    #    sub_data = data
    #end

    if !issparse(data)
        contingency_table!(X, Y, data, cont_tab)
    else
        contingency_table!(X, Y, data, cont_tab, levels_x, levels_y, nz)
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


function test{ElType <: Integer}(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{ElType}, test_name::String,
    hps::Integer, levels::AbstractVector{ElType})
    """CRITICAL: expects zeros to be trimmed from X if nz_test
    is provided!

    Test all variables Ys for univariate association with X"""
    
    levels_x = levels[X]

    if levels_x < 2
        return [TestResult(0.0, 1.0, 0, false) for Y in Ys]
    else
        max_level_y = maximum(levels[Ys])
        cont_tab = zeros(ElType, levels_x, max_level_y)
        ni = zeros(ElType, levels_x)
        nj = zeros(ElType, max_level_y)
        nz = is_zero_adjusted(test_name)

        return map(Y -> test(X, Y, data, test_name, hps, levels_x, levels[Y], cont_tab, ni, nj, nz), Ys)
    end
end

function test{ElType <: Integer}(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{ElType}, test_name::String, hps::Integer=5)
    levels = get_levels(data)

    test(X, Ys, data, test_name, hps, levels)
end

### continuous

function test{ElType <: AbstractFloat}(X::Int, Y::Int, data::AbstractMatrix{ElType}, test_name::String, n_obs_min::Integer=0,
    cor_mat::Matrix{ElType}=zeros(ElType, 0, 0), nz::Bool=false, Y_adjusted::Bool=false)

    if isempty(data)
        p_stat = 0.0
        df = 0
        pval = 1.0
        n_obs = 0
    elseif test_name == "fz" || test_name == "fz_nz"
        if isempty(cor_mat)
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
            p_stat = n_obs >= n_obs_min ? cor_mat[X, Y] : 0.0
            
        end

        df = 0
        pval = fz_pval(p_stat, n_obs, 0)
    else
        error("$test_name is not a valid test for continuous data")
    end
    TestResult(p_stat, pval, df, n_obs >= n_obs_min)
end


function test{ElType <: AbstractFloat}(X::Int, Ys::AbstractVector{Int}, data::AbstractMatrix{ElType},
        test_name::String, n_obs_min::Integer=0, cor_mat::Matrix{ElType}=zeros(ElType, 0, 0))
    """CRITICAL: expects zeros to be trimmed from X if nz_test
    is provided!
    
    Test all variables Ys for univariate association with X"""
    
    nz = is_zero_adjusted(test_name)
    
    map(Y -> test(X, Y, data, test_name, n_obs_min, cor_mat, nz, false), Ys)
end


###################
### CONDITIONAL ###
###################


function test{ElType <: AbstractFloat}(X::Int, Y::Int, Zs::AbstractVector{Int}, data::AbstractMatrix{ElType},
    test_name::String, n_obs_min::Integer, nz::Bool, cor_mat::Matrix{ElType}=zeros(ElType, 0, 0),
    pcor_set_dict::Dict{String,Dict{String,ElType}}=Dict{String,Dict{String,ElType}}(), cache_result::Bool=true)
    """Critical: expects zeros to be trimmed from both X and Y if nz is true"""

    #if needs_nz_view(Y, data, nz)
    #    sub_data = @view data[data[:, Y] .!= 0, :]
    #else
    #    sub_data = data
    #end

    if test_name == "fz" || test_name == "fz_nz"
        n_obs = size(data, 1)
        
        if n_obs >= n_obs_min
            p_stat = isempty(cor_mat) ? pcor(X, Y, Zs, data) : pcor_rec(X, Y, Zs, cor_mat, pcor_set_dict, cache_result)
            pval = fz_pval(p_stat, n_obs, 0)
        else
            p_stat = 0.0
            pval = 1.0
        end
        
        df = 0
    end
    Misc.TestResult(p_stat, pval, df, n_obs >= n_obs_min)
end


function test{ElType <: AbstractFloat}(X::Int, Y::Int, Zs::AbstractVector{Int}, data::AbstractMatrix{ElType}, test_name::String; recursive::Bool=true, n_obs_min::Integer=0)
    cor_mat = recursive ? cor(data) : zeros(ElType, 0, 0)
    pcor_set_dict = Dict{String,Dict{String,ElType}}()
    test(X, Y, Zs, data, test_name, n_obs_min, is_zero_adjusted(test_name), cor_mat, pcor_set_dict)
end


function test{ElType <: Integer}(X::Int, Y::Int, Zs::AbstractVector{Int}, data::AbstractMatrix{ElType},
        test_name::String, hps::Integer, levels_x::ElType, levels_y::ElType, cont_tab::Array{ElType,3},
    z::AbstractVector{ElType}, ni::Array{ElType,2}, nj::Array{ElType,2}, nk::Array{ElType,1}, cum_levels::AbstractVector{ElType},
    z_map_arr::AbstractVector{ElType}, nz::Bool=false,
    levels::AbstractVector{ElType}=ElType[])
    """Test association between X and Y"""

    if !issparse(data)
        levels_z = contingency_table!(X, Y, Zs, data, cont_tab, z, cum_levels, z_map_arr)
    else
        Zs_tup = Tuple(Zs)
        cont_levels = nz ? levels : nothing
        levels_z = contingency_table!(X, Y, Zs_tup, data, cont_tab, cum_levels, z_map_arr,
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


function test{ElType <: Integer}(X::Int, Y::Int, Zs::AbstractVector{Int}, data::AbstractMatrix{ElType}, test_name::String, hps::Integer=5)
    levels = get_levels(data)
    levels_x = levels[X]
    levels_y = levels[Y]
    max_levels = maximum(levels)
    max_k = length(Zs)

    max_levels_z = sum([max_levels^(i+1) for i in 1:max_k])
    cont_tab = zeros(ElType, levels_x, levels_y, max_levels_z)
    z = zeros(ElType, size(data, 1))
    ni = zeros(ElType, levels_x, max_levels_z)
    nj = zeros(ElType, levels_y, max_levels_z)
    nk = zeros(ElType, max_levels_z)
    cum_levels = zeros(ElType, max_k + 1)
    make_cum_levels!(cum_levels, Zs, levels)
    z_map_arr = zeros(ElType, max_levels_z)

    test(X, Y, Zs, data, test_name, hps, levels_x, levels_y, cont_tab, z, ni, nj, nk, cum_levels, z_map_arr, is_zero_adjusted(test_name), levels)
end


function test_subsets{ElType <: Real}(X::Int, Y::Int, Z_total::AbstractVector{Int}, data::AbstractMatrix{ElType},
    test_name::String, max_k::Integer, alpha::AbstractFloat; hps::Integer=5, n_obs_min::Integer=0, pwr::AbstractFloat=0.5,
        levels::AbstractVector{ElType}=ElType[], cor_mat::Matrix{ElType}=zeros(ElType, 0, 0),
    pcor_set_dict::Dict{String,Dict{String,ElType}}=Dict{String,Dict{String,ElType}}(), debug::Int=0, Z_wanted::AbstractVector{Int}=Int[])

    lowest_sig_result = TestResult(0.0, 0.0, 0.0, true)
    lowest_sig_Zs = Int[]
    discrete_test = isdiscrete(test_name)
    num_tests = 0
    nz = is_zero_adjusted(test_name)

    if discrete_test
        levels_x = levels[X]
        levels_y = levels[Y]
        
        max_levels = maximum(levels)
        max_levels_z = sum([max_levels^(i+1) for i in 1:max_k])
        cont_tab = zeros(ElType, levels_x, levels_y, max_levels_z)
        z = zeros(ElType, size(data, 1))
        ni = zeros(ElType, levels_x, max_levels_z)
        nj = zeros(ElType, levels_y, max_levels_z)
        nk = zeros(ElType, max_levels_z)
        cum_levels = zeros(ElType, max_k + 1)
        z_map_arr = zeros(ElType, max_levels_z)
        num_lowpwr_tests = 0
    elseif nz
        if n_obs_min > size(data, 1)
            return TestResult(0.0, 1.0, 0.0, false), Int[]
        end      
            
        empty!(pcor_set_dict)

        # compute correlations on the current subset of variables
        if !isempty(cor_mat)
            cor_subset!(data, cor_mat, [X, Y, Z_total...])
            
            if debug > 2
                println(cor_mat[[X, Y, Z_total...], [X, Y, Z_total...]])
            end
        end
    end
    
    for subset_size in max_k:-1:1
        Z_combos = isempty(Z_wanted) ? combinations(Z_total, subset_size) : combinations_with_whitelist(Z_total, Z_wanted, subset_size)
        
        for Zs in Z_combos
            if discrete_test
                make_cum_levels!(cum_levels, Zs, levels)
                test_result = test(X, Y, Zs, data, test_name, hps, levels_x, levels_y, cont_tab, z,
                                   ni, nj, nk, cum_levels, z_map_arr, nz, levels)
            else
                test_result = test(X, Y, Zs, data, test_name, n_obs_min, nz, cor_mat, pcor_set_dict, subset_size < max_k)
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
