using FlashWeave
using Test, DelimitedFiles, Statistics

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

data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])


@testset "correlation" begin
    data_clr, mask = FlashWeave.preprocess_data_default(data, "fz", verbose=false,
    prec=64)
    exp_pcor_Z1 = -0.16393307352649364
    cor_mat = cor(data_clr)
    @testset "pcor_Z1" begin
        @test isapprox(@inferred(FlashWeave.pcor(1, 16, (41,), data_clr)), exp_pcor_Z1, rtol=1e-6)
        @test isapprox(@inferred(FlashWeave.pcor_rec(1, 16, (41,), cor_mat, Dict{String,Dict{String,Float64}}())), exp_pcor_Z1, rtol=1e-6)
    end
    exp_pcor_Z3 = -0.07643814205965811
    @testset "pcor_Z3" begin
        @test isapprox(@inferred(FlashWeave.pcor(31, 21, (7, 14, 18), data_clr)), exp_pcor_Z3, rtol=1e-6)
        @test isapprox(@inferred(FlashWeave.pcor_rec(31, 21, (7, 14, 18), cor_mat, Dict{String,Dict{String,Float64}}())), exp_pcor_Z3, rtol=1e-6)
    end
    @testset "pval_fz" begin
        @test isapprox(@inferred(FlashWeave.fz_pval(exp_pcor_Z1, 351, 1)), 0.0020593283914246987, rtol=1e-6)
        @test isapprox(@inferred(FlashWeave.fz_pval(exp_pcor_Z3, 351, 3)), 0.1548665431407692, rtol=1e-6)
    end
end


@testset "mutual information" begin
    exp_mi_twoway = 0.05663301226513242
    @testset "twoway" begin
        @test isapprox(@inferred(abs(FlashWeave.mutual_information(ctab12))), exp_mi_twoway, rtol=1e-6)
    end
    @testset "threeway_Z1" begin
        @test isapprox(@inferred(FlashWeave.mutual_information(ctab12_3)), 0.0, rtol=1e-6)
    end
    @testset "threeway_Z2" begin
        @test isapprox(@inferred(FlashWeave.mutual_information(ctab12_34)), 0.0, rtol=1e-6)
    end
    @testset "pval_mi" begin
        @test isapprox(@inferred(FlashWeave.mi_pval(exp_mi_twoway, 1, 351)), 2.8770005665168745e-10, rtol=1e-6)
    end
end

@testset "FDR correction" begin
    pvals = [0.0, 1.0, 0.973774, 0.722245, 0.805758, 0.713164, 0.314595, 0.947966, 0.001, 0.0339692]
    pvals_fdr = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.786488, 1.0, 0.005, 0.113231]

    pvals_fdr_pred = copy(pvals)
    FlashWeave.benjamini_hochberg!(pvals_fdr_pred)
    @test (pvals_fdr_pred .< 0.01) == (pvals_fdr .< 0.01)
    sig_mask = pvals_fdr_pred .< 0.01
    @test isapprox(pvals_fdr_pred[sig_mask], pvals_fdr[sig_mask], rtol=1e-6)

end
