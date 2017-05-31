using Cauocc
using Base.Test

vec1 = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
vec2 = [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1]
vec3 = [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 2]
vec4 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
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

data_contingency = hcat(vec1, vec2, vec3, vec4)

for sparsity_mode in ["dense", "sparse"]
    @testset "$sparsity_mode" begin
        ts_data = sparsity_mode == "dense" ? data_contingency : sparse(data_contingency)
        @test all(Cauocc.Contingency.contingency_table(1, 2, ts_data, 2, 2) .== ctab12)
        @test all(Cauocc.Contingency.contingency_table(2, 3, ts_data, 2, 3) .== ctab23)
        @test all(Cauocc.Contingency.contingency_table(1, 2, [3], ts_data)[:,:,1:3] .== ctab12_3)
        @test all(Cauocc.Contingency.contingency_table(1, 2, [3, 4], ts_data)[:,:,1:6] .== ctab12_34)
    end
end
