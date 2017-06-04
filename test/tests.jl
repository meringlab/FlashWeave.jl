using Cauocc
using JLD
using DataFrames
using Base.Test

data = Array(readtable(joinpath("test", "data", "HMP_SRA_gut_small.tsv"))[:, 2:end])
data_clr = Cauocc.Preprocessing.preprocess_data_default(data, "fz", verbose=false, prec=64)
data_clr_nz = Cauocc.Preprocessing.preprocess_data_default(data, "fz_nz", verbose=false, prec=64)
data_bin = Cauocc.Preprocessing.preprocess_data_default(data, "mi", verbose=false, prec=64)
data_mi_nz = Cauocc.Preprocessing.preprocess_data_default(data, "mi_nz", verbose=false, prec=64)

exp_dict = load(joinpath(pwd(), "test", "data", "tests_expected.jld"))

function compare_test_results(r1::Cauocc.Misc.TestResult, r2::Cauocc.Misc.TestResult)
    isapprox(r1.stat, r2.stat, rtol=1e-6) && isapprox(r1.pval, r2.pval, rtol=1e-6) && r1.df == r2.df && r1.suff_power == r2.suff_power
end


for (test_name, data_norm) in [("mi", data_bin), ("mi_nz", data_mi_nz),
                               ("fz", data_clr), ("fz_nz", data_clr_nz)]
    @testset "$test_name" begin
        for cond_mode in ["uni", "condZ1", "condZ3"]
            @testset "$cond_mode" begin
                exp_res = exp_dict["exp_$(cond_mode)_$(test_name)"]

                if cond_mode == "uni"
                    @test all([compare_test_results(r1, r2) for (r1, r2) in zip(Cauocc.Tests.test(1, collect(2:50), data_norm, test_name), exp_res)])
                elseif cond_mode == "condZ1"
                    test_res = Cauocc.Tests.test(31, 21, [7], data_norm, test_name)
                    @test compare_test_results(test_res, exp_res)
                elseif cond_mode == "condZ3"
                    test_res = Cauocc.Tests.test(31, 21, [7, 14, 18], data_norm, test_name)
                    @test compare_test_results(test_res, exp_res)
                end

            end
        end
    end
end
