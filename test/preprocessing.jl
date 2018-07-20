using FlashWeave
using StatsBase
using FileIO
using Base.Test

data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
data_sparse = sparse(data)

exp_dict = load(joinpath("data", "preprocessing_expected.jld2"))["exp_dict"]

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
    data_norm = FlashWeave.preprocess_data(data, "rows"; make_sparse=false, verbose=false)
    @test all(isapprox.(sum(data_norm, 2), 1))
    data_norm_sparse = FlashWeave.preprocess_data(data, "rows"; make_sparse=true, verbose=false)
    @test all(data_norm .== data_norm_sparse)
end

@testset "norm per test type" begin
    for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
        @testset "$test_name" begin
            data_norm = normalize_data(data, test_name=test_name, verbose=false)

            @testset "dense" begin
                @test all(data_norm .== exp_dict[test_name])
            end

            data_norm_sparse = normalize_data(data_sparse, test_name=test_name, verbose=false)
            @testset "sparse" begin
                @test all(data_norm_sparse .== exp_dict[test_name])
            end
        end
    end
end

@testset "mi_nz fits fz_nz" begin
    data_norm_fznz = FlashWeave.preprocess_data_default(data, "fz_nz", make_sparse=false,
     verbose=false)
    data_norm_minz = FlashWeave.preprocess_data_default(data, "mi_nz", make_sparse=false, disc_method="mean",
    verbose=false)
    @test all([compare_nz_vecs(data_norm_fznz[:, i], data_norm_minz[:, i]) for i in size(data_norm_fznz, 2)])
end

# to create expected output

# data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
#
# exp_dict = Dict{String,Any}()
# for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
#     exp_dict[test_name] = normalize_data(data, test_name=test_name, verbose=false)
# end
#
# save(joinpath("data", "preprocessing_expected.jld2"), "exp_dict", exp_dict)
