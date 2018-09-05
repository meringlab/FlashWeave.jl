using FlashWeave
using FlashWeave: TestResult
using FileIO
using Test
using DelimitedFiles

data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
data_clr, mask = FlashWeave.preprocess_data_default(data, "fz", verbose=false, prec=64)
data_clr_nz, mask = FlashWeave.preprocess_data_default(data, "fz_nz", verbose=false, prec=64)
data_bin, mask = FlashWeave.preprocess_data_default(data, "mi", verbose=false, prec=64)
data_mi_nz, mask = FlashWeave.preprocess_data_default(data, "mi_nz", verbose=false, prec=64)

exp_dict = load(joinpath("data", "tests_expected.jld2"))["exp_dict"]

function compare_test_results(r1::FlashWeave.TestResult, r2::FlashWeave.TestResult)
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
                    pred_res = @inferred FlashWeave.test(1, collect(2:50), sub_data, test_name)
                    @test all([compare_test_results(r1, r2) for (r1, r2) in zip(pred_res, exp_res)])
                elseif cond_mode == "condZ1"
                    pred_res = @inferred FlashWeave.test(31, 21, (7,), sub_data, test_name)
                    @test compare_test_results(pred_res, exp_res)
                elseif cond_mode == "condZ3"
                    pred_res = @inferred FlashWeave.test(31, 21, (7, 14, 18,), sub_data, test_name)
                    @test compare_test_results(pred_res, exp_res)
                end

            end
        end
    end
end


# reproduce test results
# exp_dict = Dict{String, Any}()
# for (test_name, data_norm) in [("mi", data_bin), ("mi_nz", data_mi_nz),
#                                ("fz", data_clr), ("fz_nz", data_clr_nz)]
#     for cond_mode in ["uni", "condZ1", "condZ3"]
#         if test_name == "fz_nz"
#             if cond_mode == "uni"
#                 sub_data = @view data_norm[data_norm[:, 1] .!= 0, :]
#             else
#                 sub_data = @view data_norm[(data_norm[:, 31] .!= 0) .& (data_norm[:, 21] .!= 0), :]
#             end
#         else
#             sub_data = data_norm
#         end
#
#         if cond_mode == "uni"
#             test_res = FlashWeave.test(1, collect(2:50), sub_data, test_name)
#         elseif cond_mode == "condZ1"
#             test_res = FlashWeave.test(31, 21, (7,), sub_data, test_name)
#         elseif cond_mode == "condZ3"
#             test_res = FlashWeave.test(31, 21, (7, 14, 18,), sub_data, test_name)
#         end
#         exp_dict["exp_$(cond_mode)_$(test_name)"] = test_res
#     end
# end
# save(joinpath("data", "tests_expected.jld2"), exp_dict)
