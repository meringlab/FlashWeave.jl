using FlashWeave
using Test
using SparseArrays

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

function compare_cond_ctabs(ctab1, ctab2)
    comp_dim = min(size(ctab1, 3), size(ctab2, 3))
    ctab1 = ctab1[1:2, 1:2, 1:comp_dim]
    ctab2 = ctab2[1:2, 1:2, 1:comp_dim]

    used_js = Set{Int}()
    for i in 1:comp_dim
        slice1 = ctab1[:, :, i]
        found_hit = false
        for j in 1:comp_dim
            if !(j in used_js)
                slice2 = ctab2[:, :, j]
                if slice1 == slice2
                    found_hit = true
                    push!(used_js, j)
                    break
                end
            end
        end

        if !found_hit
            return false
        end
    end
    true
end

for sparsity_mode in ["dense", "sparse"]
    @testset "$sparsity_mode" begin
        ts_data = sparsity_mode == "dense" ? data_contingency : sparse(data_contingency)

        @testset "2-way" begin
            @test @inferred(FlashWeave.contingency_table(1, 2, ts_data, "mi"))[1:2, 1:2] == ctab12
            @test @inferred(FlashWeave.contingency_table(2, 3, ts_data, "mi"))[1:2, 1:3] == ctab23
        end

        @testset "3-way" begin
            @test compare_cond_ctabs(@inferred(FlashWeave.contingency_table(1, 2, (3,), ts_data, "mi")), ctab12_3)
            @test compare_cond_ctabs(@inferred(FlashWeave.contingency_table(1, 2, (3, 4), ts_data, "mi")), ctab12_34)
        end
    end
end
