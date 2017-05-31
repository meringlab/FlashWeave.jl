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

##################
### UNIVARIATE ###
##################

function test(X::Int, Y::Int, data::AbstractMatrix{Int}, test_name::String, hps::Int,
    levels_x::Int, levels_y::Int, cont_tab::Matrix{Int}, ni::Vector{Int}, nj::Vector{Int}, nz::Bool=false,
    data_row_inds::Vector{Int}=Int64[], data_nzero_vals::Vector{Int}=Int64[])

    if nz && (levels_y > 2)
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


function test(X::Int, Y::Int, data::AbstractMatrix{Float64}, test_name::String,
    cor_mat::Matrix{Float64}=zeros(Float64, 0, 0), nz::Bool=false)

    if nz
        sub_data = @view data[data[:, Y] .!= 0, :]
    else
        sub_data = data
    end

    if isempty(sub_data)
        p_stat = 0.0
        df = 0
        pval = 1.0
    elseif test_name == "fz" || test_name == "fz_nz"
        p_stat = isempty(cor_mat) ? cor(sub_data[:, X], sub_data[:, Y]) : cor_mat[X, Y]
        df = 0
        pval = fz_pval(p_stat, size(sub_data, 1), 0)
    else
        error("$test_name is not a valid test for continuous data")
    end
    TestResult(p_stat, pval, df, true)
end


function test(X::Int, Ys::Vector{Int}, data::AbstractMatrix{Int}, test_name::String,
    hps::Int, levels::Vector{Int}, data_row_inds::Vector{Int}=Int[], data_nzero_vals::Vector{Int}=Int[])
    """Test all variables Ys for univariate association with X"""

    levels_x = levels[X]

    if levels_x < 2
        return [TestResult(0.0, 1.0, 0, false) for Y in Ys]
    else
        max_level_y = maximum(levels[Ys])
        cont_tab = zeros(Int, levels_x, max_level_y)
        ni = zeros(Int, levels_x)
        nj = zeros(Int, max_level_y)
        nz = is_zero_adjusted(test_name)

        return map(Y -> test(X, Y, data, test_name, hps, levels_x, levels[Y], cont_tab, ni, nj, nz, data_row_inds, data_nzero_vals), Ys)
    end
end

function test(X::Int, Ys::Vector{Int}, data::AbstractMatrix{Int}, test_name::String, hps::Int=5)
    levels = get_levels(data)

    if issparse(data)
        data_row_inds = rowvals(data)
        data_nzero_vals = nonzeros(data)
    else
        data_row_inds = Int[]
        data_nzero_vals = Int[]
    end

    test(X, Ys, data, test_name, hps, levels, data_row_inds, data_nzero_vals)
end

function test(X::Int, Ys::Array{Int, 1}, data::AbstractMatrix{Float64},
        test_name::String, cor_mat::Matrix{Float64}=zeros(Float64, 0, 0))
    """Test all variables Ys for univariate association with X"""
    nz = is_zero_adjusted(test_name)
    map(Y -> test(X, Y, data, test_name, cor_mat, nz), Ys)
end


###################
### CONDITIONAL ###
###################

function test(X::Int, Y::Int, Zs::Vector{Int}, data::AbstractMatrix{Float64},
    test_name::String, cor_mat::Matrix{Float64}=zeros(Float64, 0, 0),
    pcor_set_dict::Dict{String,Dict{String,Float64}}=Dict{String,Dict{String,Float64}}(), nz::Bool=false)

    if nz
        sub_data = @view data[data[:, Y] .!= 0, :]
    else
        sub_data = data
    end

    if test_name == "fz" || test_name == "fz_nz"
        p_stat = isempty(cor_mat) ? pcor(X, Y, Zs, sub_data) : pcor_rec(X, Y, Zs, cor_mat, pcor_set_dict)
        df = 0
        pval = fz_pval(p_stat, size(sub_data, 1), 0)
    end
    Misc.TestResult(p_stat, pval, df, true)
end


function test(X::Int, Y::Int, Zs::Vector{Int}, data::AbstractMatrix{Float64}, test_name::String, recursive::Bool=true)
    cor_mat = recursive ? cor(data) : zeros(Float64, 0, 0)
    pcor_set_dict = Dict{String,Dict{String,Float64}}()
    test(X, Y, Zs, data, test_name, cor_mat, pcor_set_dict, is_zero_adjusted(test_name))
end


function test(X::Int, Y::Int, Zs::Vector{Int}, data::AbstractMatrix{Int},
        test_name::String, hps::Int, levels_x::Int, levels_y::Int, cont_tab::Array{Int,3},
    z::Vector{Int}, ni::Array{Int,2}, nj::Array{Int,2}, nk::Array{Int,1}, cum_levels::Vector{Int},
    z_map_arr::Vector{Int}, nz::Bool=false, data_row_inds::Vector{Int}=Int[], data_nzero_vals::Vector{Int}=Int[],
    levels::Vector{Int}=Int[])
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


function test(X::Int, Y::Int, Zs::Vector{Int}, data::AbstractMatrix{Int}, test_name::String, hps::Int=5)
    levels = get_levels(data)
    levels_x = levels[X]
    levels_y = levels[Y]
    max_levels = maximum(levels)
    max_k = length(Zs)

    max_levels_z = sum([max_levels^(i+1) for i in 1:max_k])
    cont_tab = zeros(Int, levels_x, levels_y, max_levels_z)
    z = zeros(Int, size(data, 1))
    ni = zeros(Int, levels_x, max_levels_z)
    nj = zeros(Int, levels_y, max_levels_z)
    nk = zeros(Int, max_levels_z)
    cum_levels = zeros(Int, max_k + 1)
    make_cum_levels!(cum_levels, Zs, levels)
    z_map_arr = zeros(Int, max_levels_z)

    if issparse(data)
        data_row_inds = rowvals(data)
        data_nzero_vals = nonzeros(data)
    else
        data_row_inds = Int[]
        data_nzero_vals = Int[]
    end

    test(X, Y, Zs, data, test_name, hps, levels_x, levels_y, cont_tab, z, ni, nj, nk, cum_levels, z_map_arr, is_zero_adjusted(test_name), data_row_inds, data_nzero_vals, levels)
end


function test_subsets{ElType <: Real}(X::Int, Y::Int, Z_total::Vector{Int}, data::AbstractMatrix{ElType},
    test_name::String, max_k::Int, alpha::Float64; hps::Int=5, pwr::Float64=0.5, levels::Vector{Int}=Int[],
    data_row_inds::Vector{Int}=Int64[], data_nzero_vals::Vector{Int}=Int64[], cor_mat::Matrix{Float64}=zeros(Float64, 0, 0),
    pcor_set_dict::Dict{String,Dict{String,Float64}}=Dict{String,Dict{String,Float64}}())

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
        cont_tab = zeros(Int, levels_x, levels_y, max_levels_z)
        z = zeros(Int, size(data, 1))
        ni = zeros(Int, levels_x, max_levels_z)
        nj = zeros(Int, levels_y, max_levels_z)
        nk = zeros(Int, max_levels_z)
        cum_levels = zeros(Int, max_k + 1)
        z_map_arr = zeros(Int, max_levels_z)
        num_lowpwr_tests = 0
    elseif nz
        empty!(pcor_set_dict)

        # compute correltions on the current subset of variables
        if !isempty(cor_mat)
            cor_subset!(data, cor_mat, [X, Y, Z_total...])
        end
    end

    for subset_size in 1:max_k
        Z_combos = combinations(Z_total, subset_size)

        for Zs in Z_combos
            if discrete_test
                make_cum_levels!(cum_levels, Zs, levels)
                test_result = test(X, Y, Zs, data, test_name, hps, levels_x, levels_y, cont_tab, z,
                                   ni, nj, nk, cum_levels, z_map_arr, nz, data_row_inds, data_nzero_vals, levels)
            else
                test_result = test(X, Y, Zs, data, test_name, cor_mat, pcor_set_dict, nz)
            end
            num_tests += 1

            # if discrete test didn't have enough power, check if
            # the threshold of number of unreliable tests has been reached
            if discrete_test & !test_result.suff_power
                num_lowpwr_tests += 1

                if num_lowpwr_tests / num_tests >= 1 - pwr
                    lowest_sig_result.suff_power = false
                    return lowest_sig_result, Zs
                end
            else
                if !issig(test_result, alpha)
                    return test_result, Zs
                elseif test_result.pval > lowest_sig_result.pval
                    lowest_sig_result = test_result
                    lowest_sig_Zs = Zs
                end
            end
        end
    end

    lowest_sig_result, lowest_sig_Zs
end

end
