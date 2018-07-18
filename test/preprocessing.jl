using FlashWeave
using StatsBase
using JLD2, FileIO
using Base.Test

data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])

exp_dict = load(joinpath("data", "preprocessing_expected.jld"))["exp_dict"]

function compare_nz_vecs(fznz_vec, minz_vec, verbose=false)
    fznz_vec_red = fznz_vec[fznz_vec .!= 0]
    minz_vec_red = minz_vec[minz_vec .!= 0]
    if length(fznz_vec_red) != length(minz_vec_red)
        verbose && println("Nonzero elements do not match")
        return false
    else
        fznz_pos_mask = fznz_vec_red .> mean(fznz_vec_red)
        minz_pos_mask = minz_vec_red .== 2
        if !all(fznz_pos_mask .== minz_pos_mask)
            verbose && println("Above mean elements do not fit")
            return false
        else
            return true
        end
    end
end

@testset "TSS" begin
    data_norm = FlashWeave.Preprocessing.preprocess_data(data, "rows"; make_sparse=false, verbose=false)
    @test all(isapprox.(sum(data_norm, 2), 1))
    data_norm_sparse = FlashWeave.Preprocessing.preprocess_data(data, "rows"; make_sparse=true, verbose=false)
    @test all(data_norm .== data_norm_sparse)
end

@testset "norm per test type" begin
    for test_name in ["mi", "mi_nz", "fz", "fz_nz"]#, "fzr", "fzr_nz"] ## if you want to test ranked fz
        @testset "$test_name" begin
            if startswith(test_name, "fzr")
                test_name = replace(test_name, "fzr", "fz")
                rank_clr = true
            else
                rank_clr = false
            end

            data_norm = FlashWeave.Preprocessing.preprocess_data_default(data, test_name; make_sparse=false,
             verbose=false, rank_clr=rank_clr)

            if !rank_clr
                @test all(data_norm .== exp_dict[test_name])
            end

            data_norm_sparse = FlashWeave.Preprocessing.preprocess_data_default(data, test_name; make_sparse=true,
             verbose=false, rank_clr=rank_clr)
            @test all(data_norm .== data_norm_sparse)
        end
    end
end

@testset "mi_nz fits fz_nz" begin
    data_norm_fznz = FlashWeave.Preprocessing.preprocess_data_default(data, "fz_nz", make_sparse=false,
     verbose=false)
    data_norm_minz = FlashWeave.Preprocessing.preprocess_data_default(data, "mi_nz", make_sparse=false, disc_method="mean",
    verbose=false)
    @test all([compare_nz_vecs(data_norm_fznz[:, i], data_norm_minz[:, i]) for i in size(data_norm_fznz, 2)])
end

# to create expected output

# data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
#
#
# exp_dict = Dict{String,Any}()
# for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
#     exp_dict[test_name] = FlashWeave.Preprocessing.preprocess_data_default(data, test_name, make_sparse=false)
# end
#
# save(joinpath("data", "preprocessing_expected.jld"), "exp_dict", exp_dict)
