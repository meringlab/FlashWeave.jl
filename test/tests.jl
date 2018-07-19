using FlashWeave
using JLD2, FileIO
using Base.Test

data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
data_clr = FlashWeave.Preprocessing.preprocess_data_default(data, "fz", verbose=false, prec=64)
data_clr_nz = FlashWeave.Preprocessing.preprocess_data_default(data, "fz_nz", verbose=false, prec=64)
data_bin = FlashWeave.Preprocessing.preprocess_data_default(data, "mi", verbose=false, prec=64)
data_mi_nz = FlashWeave.Preprocessing.preprocess_data_default(data, "mi_nz", verbose=false, prec=64)

exp_dict = load(joinpath("data", "tests_expected.jld2"))

function compare_test_results(r1::FlashWeave.Types.TestResult, r2::FlashWeave.Types.TestResult)
    isapprox(r1.stat, r2.stat, rtol=1e-2) && isapprox(r1.pval, r2.pval, rtol=1e-2) && r1.df == r2.df && r1.suff_power == r2.suff_power
end


for (test_name, data_norm) in [("mi", data_bin), ("mi_nz", data_mi_nz),
                               ("fz", data_clr), ("fz_nz", data_clr_nz)]
    @testset "$test_name" begin
        for cond_mode in ["uni", "condZ1", "condZ3"]
            @testset "$cond_mode" begin
                exp_res = exp_dict["exp_$(cond_mode)_$(test_name)"]

                if test_name == "fz_nz"
                    if cond_mode == "uni"
                        sub_data = @view data_norm[data_norm[:, 1] .!= 0, :]
                    else
                        sub_data = @view data_norm[(data_norm[:, 31] .!= 0) .& (data_norm[:, 21] .!= 0), :]
                    end
                else
                    sub_data = data_norm
                end

                if cond_mode == "uni"
                    @test all([compare_test_results(r1, r2) for (r1, r2) in zip(FlashWeave.Tests.test(1, collect(2:50), sub_data, test_name), exp_res)])
                elseif cond_mode == "condZ1"
                    test_res = FlashWeave.Tests.test(31, 21, (7,), sub_data, test_name)
                    @test compare_test_results(test_res, exp_res)
                elseif cond_mode == "condZ3"
                    test_res = FlashWeave.Tests.test(31, 21, (7, 14, 18,), sub_data, test_name)
                    @test compare_test_results(test_res, exp_res)
                end

            end
        end
    end
end
