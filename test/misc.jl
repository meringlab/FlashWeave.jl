using FlashWeave
using Test

@testset "precision_conversion" begin
    A_base64 = vcat(ones(100, 10), zeros(100, 10))
    A_base32 = Matrix{Float32}(A_base64)
    for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
        @testset "$(test_name)" begin
            for prec in [32, 64]
                @testset "" begin
                    for (base_prec, A) in [("32", A_base32), ("64", A_base64)]
                        @testset "target/base prec: $(prec) $(base_prec)" begin
                            for make_sparse in [true, false]
                                @testset "sparse_$(make_sparse)" begin
                                    A_conv = FlashWeave.convert_to_target_prec(A, prec, make_sparse; test_name=test_name)
                                    @test endswith(string(eltype(A_conv)), string(prec))
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end