using JLD2, FileIO
using Base.Test
using SimpleWeightedGraphs
using LightGraphs
using FlashWeave

data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
adj_exp_dict = load(joinpath("data", "learning_expected.jld2"))

exp_dict = Dict(key=>SimpleWeightedGraph(adj_mat) for (key, adj_mat) in adj_exp_dict)


function make_network(data, test_name, make_sparse=false, prec=64, verbose=false; kwargs...)
    data_norm = FlashWeave.preprocess_data_default(data, test_name, verbose=false, make_sparse=make_sparse, prec=prec)
    kwargs_dict = Dict(kwargs)
    graph_res = FlashWeave.LGL(data_norm; test_name=test_name, verbose=verbose,  kwargs...)
    graph_res.graph
end


function compare_graph_results(g1::SimpleWeightedGraph, g2::SimpleWeightedGraph; verbose=false, rtol=1e-2, atol=0.0, approx=false, approx_nbr_diff=1, approx_weight_meandiff=0.15)
    if nv(g1) != nv(g2)
        verbose && println("Nodes don't match")
        return false
    end

    nbr_diff = 0
    weight_diffnum = 0
    weight_diffvec = Float64[]
    for T in vertices(g1)
        nbrs_g1 = Set(neighbors(g1, T))
        nbrs_g2 = Set(neighbors(g2, T))

        if nbrs_g1 != nbrs_g2
            num_diff_nbrs = length(symdiff(nbrs_g1, nbrs_g2))
            verbose && println("Neighbors for node $T dont match ($(num_diff_nbrs) differ)")

            if approx
                nbr_diff += num_diff_nbrs
            end

            if !approx || nbr_diff > approx_nbr_diff
                return false
            end
        end

        for nbr in intersect(nbrs_g1, nbrs_g2)
            g1_weight, g2_weight = [G.weights[T, nbr] for G in [g1, g2]]
            if !isapprox(g1_weight, g2_weight, rtol=rtol, atol=atol)
                verbose && println("Weights for node $T and neighbor $nbr dont fit: $(g1_weight), $(g2_weight)")

                if approx
                    rel_diff = abs(g1_weight - g2_weight) / max(g1_weight, g2_weight)
                    push!(weight_diffvec, rel_diff)
                else
                    return false
                end
            end
        end
    end

    if approx
        if !isempty(weight_diffvec)
            weight_meandiff = mean(weight_diffvec)

            if weight_meandiff > approx_weight_meandiff
                verbose && println("Relative difference between mismatched weights $(weight_meandiff) > $(approx_weight_meandiff)")
                return false
            else
                return true
            end
        else
            return true
        end
    else
        return true
    end
end


@testset "main_test_modes" begin
    for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
        @testset "$test_name" begin
            for max_k in [0, 3]
                @testset "max_k $max_k" begin
                    exp_graph = exp_dict["exp_$(test_name)_maxk$(max_k)"]

                    for make_sparse in [true, false]
                        @testset "sparse $make_sparse" begin
                            for parallel in ["single", "single_il", "multi_il"]

                                if max_k == 0 && parallel == "single_il"
                                    continue
                                end


                                @testset "parallel $parallel" begin
                                    is_il = endswith(parallel, "_il")
                                    time_limit = is_il ? 30.0 : 0.0
                                    graph = make_network(data, test_name, make_sparse, 64,
                                        max_k=max_k, parallel=parallel, time_limit=time_limit)


                                    rtol = 1e-2
                                    atol = 0.0

                                    # special case for conditional mi
                                    if is_il && test_name == "mi" && max_k == 3
                                        approx_nbr_diff = 22
                                        approx_weight_meandiff = 0.16
                                    else
                                        approx_nbr_diff = 0
                                        approx_weight_meandiff = 0.05
                                    end

                                    @testset "edge_identity" begin
                                        @test compare_graph_results(exp_graph, graph,
                                                                    rtol=rtol, atol=atol,
                                                                    approx=is_il,
                                                                    approx_nbr_diff=approx_nbr_diff,
                                                                    approx_weight_meandiff=approx_weight_meandiff)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


@testset "precision_32" begin
    for (test_name, make_sparse) in [("mi_nz", true), ("fz", false)]
        for make_sparse in [true]
            @testset "$(test_name)_$(make_sparse)_single" begin
                graph = make_network(data, test_name, make_sparse, 32, max_k=3, parallel="single", time_limit=0.0)
                exp_graph = exp_dict["exp_$(test_name)_maxk3"]
                @test compare_graph_results(exp_graph, graph, rtol=1e-2, atol=0.0)
            end
        end
    end
end


# to create expected output

# exp_dict = Dict()
# for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
#     for max_k in [0, 3]
#         graph = make_network(data, test_name, false, 64, max_k=max_k, parallel="single", time_limit=0.0)
#         exp_dict["exp_$(test_name)_maxk$(max_k)"] = graph.weights
#      end
# end
#
# out_path = joinpath("data", "learning_expected.jld")
# rm(out_path)
# save(out_path, exp_dict)
