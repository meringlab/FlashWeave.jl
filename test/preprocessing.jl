using FlashWeave
using StatsBase
using Base.Test

#data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
#data_sparse = sparse(data)
#exp_dict = load(joinpath("data", "preprocessing_expected.jld"))

for disc_type in ["continuous", "discrete"]
    @testset disc_type begin
        for (norm_mode, norm_fun) in norm_dict[disc_type]
            @testset norm_mode begin
                #data_exp = exp_dict[disc_type][norm_mode]
                for sparsity in [true, false]
                    @testset "sparse $sparsity" begin
                    #data_curr = sparsity ? data_sparse : data
                    #data_norm = norm_fun(data_curr)
                    #@test all(data_norm .== data_exp)
                end
            end
        end
    end
end


### Gold-standard generation part
#data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
#data_rownorm = data ./ sum(data, 2)
#@assert all(isapprox.(sum(data_rownorm, 2), 1))

#data_clrnorm =
