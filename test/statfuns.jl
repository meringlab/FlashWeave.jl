using Cauocc
using Base.Test

@testset "correlation" begin
end

@testset "mutual information" begin
end

@testset "FDR correction" begin
    const pvals = [0.0,1.0,0.973774,0.722245,0.805758,0.713164,
    0.314595,0.947966,0.05,0.0339692]
    const pvals_fdr = [0.0,1.0,1.0,1.0,1.0,1.0,0.786487,1.0,0.166667,0.166667]
    @test isapprox(Cauocc.Statfuns.benjamini_hochberg(pvals), pvals_fdr, rtol=0.001)
end
