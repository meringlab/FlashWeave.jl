## tests.jl

#





# conversion to new test results (switch from cauocc to flashweave, but also usable for later updates)
flashw_test_res(cauocc_test_res) = FlashWeave.Misc.TestResult(cauocc_test_res.stat, cauocc_test_res.pval, cauocc_test_res.df, cauocc_test_res.suff_power)

function convert_exp_dict(exp_dict)
    new_exp_dict = Dict{String,Any}()
    for (key, val) in exp_dict
        if isa(val, Array)
            new_exp_dict[key] = map(flashw_test_res, val)
        else
            new_exp_dict[key] = flashw_test_res(val)
        end
    end
    new_exp_dict
end
