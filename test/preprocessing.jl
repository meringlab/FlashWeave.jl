using FlashWeave
using StatsBase
using FileIO
using Test

data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
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
    data_norm, mask = FlashWeave.preprocess_data(data, "rows"; make_sparse=false, verbose=false)
    @test all(isapprox.(sum(data_norm, 2), 1))
    data_norm_sparse, mask = FlashWeave.preprocess_data(data, "rows"; make_sparse=true, verbose=false)
    @test all(data_norm .== data_norm_sparse)
end

@testset "norm per test type" begin
    for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
        @testset "$test_name" begin
            data_norm, mask = normalize_data(data, test_name=test_name, verbose=false)

            @testset "dense" begin
                @test all(data_norm .== exp_dict[test_name])
            end

            data_norm_sparse, mask = normalize_data(data_sparse, test_name=test_name, verbose=false)
            @testset "sparse" begin
                @test all(data_norm_sparse .== exp_dict[test_name])
            end
        end
    end
end

@testset "filter data" begin
    @testset "zero counts" begin
        wanted_zero_otus = 20
        wanted_binfilt_otus = 10
        added_zero_samples = 10
        binfilt_otu_data = vcat(zeros(eltype(data), (size(data, 1) - 1, wanted_binfilt_otus)),
                                ones(eltype(data), (1, wanted_binfilt_otus)))
        rm_data = hcat(data, binfilt_otu_data)

        zerocount_otu_data = zeros(eltype(data), (size(data, 1), wanted_zero_otus))
        rm_data = hcat(rm_data, zerocount_otu_data)

        rm_data = vcat(rm_data, zeros(eltype(data), (added_zero_samples, size(rm_data, 2))))
        wanted_zero_samples = added_zero_samples + 5 # some zero samples already present in data
        rm_data_sparse = sparse(rm_data)
        rm_header = map(string, 1:size(rm_data, 2))
        meta_mask = falses(length(rm_header))
        wanted_header_zero = rm_header[1:size(data, 2)+wanted_binfilt_otus]
        wanted_header_binfilt = rm_header[1:size(data, 2)]

        for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
            @testset "$test_name" begin
                data_norm, header_norm, mask = normalize_data(rm_data, test_name=test_name, header=rm_header,
                                                         meta_mask=meta_mask, verbose=false)

                for (zero_desc, zero_dim, zero_count) in [("OTUs", 2, wanted_zero_otus),
                                                          ("samples", 1, wanted_zero_samples)]
                    @testset "$zero_desc" begin
                        if test_name == "mi_nz" && zero_desc == "OTUs"
                            zero_count += wanted_binfilt_otus
                        end
                        @test size(data_norm, zero_dim) == size(rm_data, zero_dim) - zero_count

                        if zero_desc == "OTUs"
                            wanted_header = test_name == "mi_nz" ? wanted_header_binfilt : wanted_header_zero
                            @testset "header" begin
                                @test header_norm == wanted_header
                            end
                        end
                    end
                end
            end
        end
    end
end


@testset "mi_nz fits fz_nz" begin
    data_norm_fznz, mask = FlashWeave.preprocess_data_default(data, "fz_nz", make_sparse=false,
     verbose=false)
    data_norm_minz, mask = FlashWeave.preprocess_data_default(data, "mi_nz", make_sparse=false, disc_method="mean",
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
