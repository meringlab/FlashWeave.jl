using Cauocc
using DataFrames
using Base.Test

const data = Array(readtable(joinpath("test", "data", "HMP_SRA_gut_small.tsv"))[:, 2:end])

const exp_num_nbr_dict = Dict("mi" => Dict(0 => [24,6,5,14,5,14,16,10,6,6,8,13,4,15,23,3,4,8,8,
                                                 13,4,4,22,3,7,12,6,14,11,16,18,11,17,8,6,6,1,
                                                 2,12,2,20,9,10,19,5,1,11,9,7,16],
                                           3 => [10,4,1,4,3,6,6,2,3,2,4,5,2,7,6,2,3,2,3,3,2,
                                                 3,5,2,2,3,2,2,3,5,8,3,5,3,3,3,1,2,3,2,4,3,5,7,
                                                 3,1,4,5,2,4]),
                              "mi_nz" => Dict(0 => [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
                                                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
                                                    0,0,0,0,0,0],
                                              3 => [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
                                                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
                                                    0,0,0,0,0,0,0,0]),
                              "fz" => Dict(0 => [2,2,2,5,1,1,8,1,2,0,3,1,1,7,0,5,1,3,1,1,6,
                                                 2,1,1,1,3,1,3,3,1,7,1,3,1,5,2,3,0,2,0,8,3,
                                                 0,3,1,1,4,1,4,2],
                                           3 => [2,2,2,3,1,1,6,1,2,0,3,1,1,5,0,5,1,2,1,1,4,2,
                                                 1,1,1,2,1,3,3,1,3,1,2,1,4,2,3,0,2,0,5,3,0,3,
                                                 1,1,3,1,4,2]),
                              "fz_nz" => Dict(0 => [1,0,0,1,0,0,4,0,0,0,0,1,0,0,0,1,0,2,0,0,1,
                                                    0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,4,0,
                                                    0,1,0,0,0,0,0,0],
                                              3 => [1,0,0,1,0,0,4,0,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,
                                                    0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,3,0,0,1,0,0,
                                                    0,0,0,0]))


function make_network(data, test_name, make_sparse=false; kwargs...)
    data_norm = Cauocc.Preprocessing.preprocess_data_default(data, test_name, verbose=false, make_sparse=make_sparse)
    graph_dict = LGL(data_norm; test_name=test_name, verbose=false, kwargs...)
end

function get_num_nbr(data, test_name, make_sparse=false; kwargs...)
    graph_dict = make_network(data, test_name, make_sparse; kwargs...)
    map(length, [graph_dict[key] for key in sort(collect(keys(graph_dict)))])
end

@testset "single" begin
    for (test_name, sub_dict) in exp_num_nbr_dict
        @testset "$test_name" begin
            for (max_k, exp_num_nbr) in sub_dict
                @testset "max_k $max_k" begin
                    for make_sparse in [true, false]
                        @testset "sparse $make_sparse" begin
                            @test all(get_num_nbr(data, test_name, make_sparse, max_k=max_k, parallel="single") .== exp_num_nbr)
                        end
                    end
                end
            end
        end
    end
end

"""
@testset "parallel" begin
    for (test_name, sub_dict) in exp_num_nbr_dict
        for (max_k, exp_num_nbr) in sub_dict
            for make_sparse in [true, false]
                num_diffs = get_num_nbr(data, test_name, make_sparse, max_k=max_k, parallel="multi_il", time_limit=0.0) .-
                            exp_num_nbr |> abs |> sum
                @test (test_name == "mi" && num_diffs == 20) || num_diffs == 0
            end
        end
    end
end
"""
