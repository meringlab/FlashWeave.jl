using Cauocc
using DataFrames
using Base.Test

ctab12 = [4 2; 2 4]
ctab23 = [6 0 0; 0 5 1]
ctab12_3 = zeros(Int, 2, 2, 3)
ctab12_3[1, 1, 1] = 4
ctab12_3[2, 1, 1] = 2
ctab12_3[1, 2, 2] = 2
ctab12_3[2, 2, 2] = 3
ctab12_3[2, 2, 3] = 1
ctab12_34 = zeros(Int, 2, 2, 6)
ctab12_34[1, 1, 1] = 2
ctab12_34[1, 2, 2] = 2
ctab12_34[2, 2, 2] = 2
ctab12_34[1, 1, 3] = 2
ctab12_34[2, 1, 3] = 2
ctab12_34[2, 2, 4] = 1
ctab12_34[2, 2, 5] = 1

data = Array(readtable(joinpath("test", "data", "HMP_SRA_gut_small.tsv"))[:, 2:end])

@testset "correlation" begin
    data_clr = Cauocc.Preprocessing.preprocess_data_default(data, "fz", verbose=false)
    exp_pcor_Z1 = -0.16393307352649364
    cor_mat = cor(data_clr)
    @testset "pcor_Z1" begin
        @test isapprox(Cauocc.Statfuns.pcor(1, 16, [41], data_clr), exp_pcor_Z1, rtol=1e-6)
        @test isapprox(Cauocc.Statfuns.pcor_rec(1, 16, [41], cor_mat, Dict{String,Dict{String,Float64}}()), exp_pcor_Z1, rtol=1e-6)
    end
    exp_pcor_Z3 = -0.07643814205965811
    @testset "pcor_Z3" begin
        @test isapprox(Cauocc.Statfuns.pcor(31, 21, [7, 14, 18], data_clr), exp_pcor_Z3, rtol=1e-6)
        @test isapprox(Cauocc.Statfuns.pcor_rec(31, 21, [7, 14, 18], cor_mat, Dict{String,Dict{String,Float64}}()), exp_pcor_Z3, rtol=1e-6)
    end
    @testset "pval_fz" begin
        @test isapprox(Cauocc.Statfuns.fz_pval(exp_pcor_Z1, 351, 1), 0.0020593283914246987, rtol=1e-6)
        @test isapprox(Cauocc.Statfuns.fz_pval(exp_pcor_Z3, 351, 3), 0.1548665431407692, rtol=1e-6)
    end
end


@testset "mutual information" begin
    exp_mi_twoway = 0.05663301226513242
    @testset "twoway" begin
        @test isapprox(Cauocc.Statfuns.mutual_information(ctab12), exp_mi_twoway, rtol=1e-6)
    end
    # threeway tests should be improved to incorporate non-zero mi
    @testset "threeway_Z1" begin
        @test isapprox(Cauocc.Statfuns.mutual_information(ctab12_3), 0.0, rtol=1e-6)
    end
    @testset "threeway_Z2" begin
        @test isapprox(Cauocc.Statfuns.mutual_information(ctab12_34), 0.0, rtol=1e-6)
    end
    @testset "pval_mi" begin
        @test isapprox(Cauocc.Statfuns.mi_pval(exp_mi_twoway, 1, 351), 2.8770005665168745e-10, rtol=1e-6)
    end
end

@testset "FDR correction" begin
    pvals = [0.0,1.0,0.973774,0.722245,0.805758,0.713164,
    0.314595,0.947966,0.05,0.0339692]
    pvals_fdr = [0.0,1.0,1.0,1.0,1.0,1.0,0.786487,1.0,0.166667,0.166667]
    @test isapprox(Cauocc.Statfuns.benjamini_hochberg(pvals), pvals_fdr, rtol=1e-6)
end
