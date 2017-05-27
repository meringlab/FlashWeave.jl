using Cauocc
using DataFrames
using Base.Test

#const data_path joinpath("data", "HMP_SRA_gut_small.tsv")]
const data = Array(readtable(joinpath("data", "HMP_SRA_gut_small.tsv"))[:, 2:end])

const exp_num_nbr_dict = Dict("mi" => Dict(0 => [24,6,5,14,5,14,16,10,6,6,8,13,4,15,23,3,4,8,8,13,4,4,22,3,7,
                                                 12,6,14,11,16,18,11,17,8,6,6,1,2,12,2,20,9,10,19,5,1,11,9,7,16],
                                           3 => [10,4,1,4,3,6,6,2,3,2,4,5,2,7,6,2,3,2,3,3,2,3,5,2,2,3,2,2,3,5,8,
                                                 3,5,3,3,3,1,2,3,2,4,3,5,7,3,1,4,5,2,4]),
                              "mi_nz" => Dict(0 => [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                                    0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                              3 => [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                                    0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
                              "fz" => Dict(0 => [2,2,2,5,1,1,8,1,2,0,3,1,1,7,0,5,1,3,1,1,6,2,1,1,1,3,1,3,3,1,7,1,3,
                                                 1,5,2,3,0,2,0,8,3,0,3,1,1,4,1,4,2],
                                           3 => [2,2,2,3,1,1,6,1,2,0,3,1,1,5,0,5,1,2,1,1,4,2,1,1,1,2,1,3,3,1,3,1,2,
                                                 1,4,2,3,0,2,0,5,3,0,3,1,1,3,1,4,2]),
                              "fz_nz" => Dict(0 => [1,0,0,1,0,0,4,0,0,0,0,1,0,0,0,1,0,2,0,0,1,0,0,0,0,0,0,0,0,0,1,
                                                    0,0,0,0,0,1,0,0,0,4,0,0,1,0,0,0,0,0,0],
                                              3 => [1,0,0,1,0,0,4,0,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,
                                                    0,0,0,0,0,1,0,0,0,3,0,0,1,0,0,0,0,0,0]))


function get_num_nbr(data, test_name; kwargs...)
  data_norm = Cauocc.Preprocessing.preprocess_data_default(data, test_name, verbose=false)
  graph_dict = LGL(data_norm; test_name=test_name, verbose=false, edge_rule="OR", kwargs...)
  map(length, [graph_dict[key] for key in 1:size(data_norm, 2)])
end

@testset "learning_single" begin

  for (test_name, sub_dict) in exp_num_nbr_dict
    for (max_k, exp_num_nbr) in sub_dict
      @test all(get_num_nbr(data, test_name, max_k=max_k, parallel="single") .== exp_num_nbr)
    end
  end

end

@testset "learning_parallel" begin
  for (test_name, sub_dict) in exp_num_nbr_dict
    for (max_k, exp_num_nbr) in sub_dict
      num_diffs = get_num_nbr(data, test_name, max_k=max_k, parallel="multi_il", time_limit=0.0) .- exp_num_nbr |> abs |> sum
      println(num_diffs)
      @test (test_name == "mi" && num_diffs == 20) || num_diffs == 0
    end
  end
end
