flashw_test_res(cauocc_test_res) = FlashWeave.Misc.TestResult(cauocc_test_res.stat, cauocc_test_res.pval, cauocc_test_res.df, cauocc_test_res.suff_power)

function convert_exp_dict(exp_dict)
    new_exp_dict = Dict{String,Any}()
    for (key, val) in exp_dict
        if isa(val, Array)
            new_exp_dict[key] =


     = Dict([(x, isa(y, Array) ? y : FlashWeave.Misc.TestResult(y.stat, y.pval, y.df, y.suff_power)) for (x, y) in exp_dict])
