using FlashWeave
using StatsBase
using FileIO
using Test
using SparseArrays, DelimitedFiles, Statistics

data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
data_sparse = sparse(data)

exp_dict = load(joinpath("data", "preprocessing_expected.jld2"), "exp_dict")

function compare_nz_vecs(fznz_vec, minz_vec, verbose=false)
    fznz_vec_red = fznz_vec[fznz_vec .!= 0]
    minz_vec_red = minz_vec[minz_vec .!= 0]
    if length(fznz_vec_red) != length(minz_vec_red)
        verbose && println("Nonzero elements do not match")
        return false
    else
        fznz_pos_mask = fznz_vec_red .> mean(fznz_vec_red)
        minz_pos_mask = minz_vec_red .== 2
        if fznz_pos_mask != minz_pos_mask
            verbose && println("Above mean elements do not fit")
            return false
        else
            return true
        end
    end
end


@testset "clr_adapt eps" begin
    s1 = vcat([10000.0 for i in 1:10000], zeros(10))
    s2 = vcat([100.0 for i in 1:10], zeros(10000))
    s3 = collect(1:10010)
    mat = permutedims(hcat(s1, s2, s3))
    mat_norm = normalize_data(mat, test_name="fz", verbose=false)[1]
    @test all(isfinite.(mat_norm))
    @test size(mat_norm, 1) == 2
end


@testset "norm per test type" begin
    for norm_pair in [("clr-adapt", "fz"), ("clr-nonzero", "fz_nz"),
                                   ("clr-nonzero-binned", "mi_nz"), ("pres-abs", "mi"),
                                   ("tss", ""), ("tss-nonzero-binned", "")]
        data_norm_exp = exp_dict[norm_pair[1]]
        for (i, norm_desc) in enumerate(norm_pair)
            norm_mode = test_name = ""
            if norm_desc != ""
                @testset "$norm_desc" begin
                    if i == 1
                        norm_mode = norm_desc
                    else
                        test_name = norm_desc
                    end
                    data_norms = [normalize_data(curr_data, norm_mode=norm_mode, test_name=test_name,
                                                 verbose=false)[1]
                                  for curr_data in (data, data_sparse)]

                    @testset "dense" begin
                        @test data_norms[1] == data_norm_exp
                    end

                    @testset "sparse" begin
                        @test data_norms[2] == data_norm_exp
                    end
                end
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
    data_norm_fznz, _, mask = FlashWeave.preprocess_data_default(data, "fz_nz", make_sparse=false,
                                                                 verbose=false)
    data_norm_minz, _, mask = FlashWeave.preprocess_data_default(data, "mi_nz", make_sparse=false,
                                                                 disc_method="mean", verbose=false)
    @test all([compare_nz_vecs(data_norm_fznz[:, i], data_norm_minz[:, i]) for i in size(data_norm_fznz, 2)])
end


@testset "meta data" begin
    @testset "one-hot" begin
        data_threecat, header_threecat = readdlm(joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_tiny.tsv"), '\t', header=true)
        header_threecat = Vector{String}(header_threecat[:])
        meta_data_threecat, meta_header_threecat = readdlm(joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_tiny_meta_oneHotTest.tsv"), '\t', header=true)
        meta_header_threecat = Vector{String}(meta_header_threecat[:])
        data_conc = hcat(data_threecat, meta_data_threecat)
        header_conc = vcat(header_threecat, meta_header_threecat)
        meta_mask = BitVector(vcat(zeros(length(header_threecat)), ones(length(meta_header_threecat))))
        for test_name in ["fz", "mi", "fz_nz", "mi_nz"]
            @testset "$test_name" begin
                for make_onehot in [true, false]
                    @testset "$make_onehot" begin
                        data_norm, header_norm, meta_mask_norm, row_mask = FlashWeave.preprocess_data_default(data_conc, test_name, make_sparse=false,
                        verbose=false, header=header_conc, meta_mask=meta_mask, make_onehot=make_onehot)

                        if make_onehot
                            # skip the continuous column for identity test
                            @test data_norm[:, meta_mask_norm][:, 1:end-1] == exp_dict["meta_tiny_oneHotTest"].meta_data[row_mask, 1:end-1]
                            @test header_norm[meta_mask_norm] == exp_dict["meta_tiny_oneHotTest"].meta_header

                            # check the continuous meta data column
                            if startswith(test_name, "mi")
                                @test length(unique(data_norm[:, end])) == 2
                            end
                        else
                            # smoke test for onehot == false
                            @test sum(meta_mask_norm) == sum(meta_mask)
                            @test size(data_norm, 2) == length(header_norm)
                        end
                    end
                end
            end
        end
    end
end

# to create expected output

# data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
#
# exp_dict = Dict{String,Any}()
# for (norm_mode, test_name) in [("clr-adapt", "fz"), ("clr-nonzero", "fz_nz"),
#                   ("clr-nonzero-binned", "mi_nz"), ("pres-abs", "mi"),
#                   ("tss", ""), ("tss-nonzero-binned", "")]
#     data_norm = normalize_data(data, norm_mode=norm_mode, verbose=false)[1]
#     if test_name != ""
#         @assert normalize_data(data, test_name=test_name, verbose=false)[1] == data_norm
#     end
#     exp_dict[norm_mode] = data_norm
# end
#
# save(joinpath("data", "preprocessing_expected.jld2"), "exp_dict", exp_dict)
