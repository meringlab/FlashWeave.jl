using FlashWeave
using SimpleWeightedGraphs
using JLD2, FileIO
using Base.Test

graph = load(joinpath("data", "misc_expected.jld"))["graph"]

@testset "IO" begin
    tmp_path = tempname()
    FlashWeave.Misc.write_edgelist(tmp_path, graph)
    graph_el = FlashWeave.Misc.read_edgelist(tmp_path)
    @test graph == graph_el
end


@testset "weights" begin
end


# to create expected output

# max_k = 3
# make_sparse = false
# parallel = "single"
# test_name = "mi"
# graph = make_network(data, test_name, make_sparse, 64, true, max_k=max_k, parallel=parallel, time_limit=30.0, correct_reliable_only=false, n_obs_min=0, debug=0, verbose=true, FDR=true, weight_type="cond_stat") # make_network: see test/learning.jl
#save(joinpath("data", "misc_expected.jld"), "graph", graph)
