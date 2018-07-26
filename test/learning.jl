using Base.Test
using SimpleWeightedGraphs
using LightGraphs

nprocs() == 1 && addprocs(1)
using FlashWeave
using FileIO

data_path = joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_small.tsv")
data = Matrix{Float64}(readdlm(data_path, '\t')[2:end, 2:end])
data_sp = sparse(data)

adj_exp_dict = load(joinpath("data", "learning_expected.jld2"))

exp_dict = Dict(key=>SimpleWeightedGraph(adj_mat) for (key, adj_mat) in adj_exp_dict)

rtol = 1e-2
atol = 0.0

function make_network(data, test_name, make_sparse=false, prec=64, verbose=false; kwargs...)
    data_norm = FlashWeave.preprocess_data_default(data, test_name, verbose=false, make_sparse=make_sparse, prec=prec)
    kwargs_dict = Dict(kwargs)
    lgl_res = FlashWeave.LGL(data_norm; test_name=test_name, verbose=verbose,  kwargs...)
    lgl_res.graph
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


@testset "LGL_backend" begin
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
                                    pred_graph = make_network(data, test_name, make_sparse, 64,
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
                                        @test compare_graph_results(exp_graph, pred_graph,
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
        for make_sparse in [true, false]
            @testset "$(test_name)_$(make_sparse)_single" begin
                pred_graph = make_network(data, test_name, make_sparse, 32, max_k=3, parallel="single", time_limit=0.0)
                exp_graph = exp_dict["exp_$(test_name)_maxk3"]
                @test compare_graph_results(exp_graph, pred_graph, rtol=1e-2, atol=0.0)
            end
        end
    end
end

@testset "learn_network" begin
    approx_nbr_diff = 0
    approx_weight_meandiff = 0.05
    header = ["X" * string(i) for i in 1:size(data, 2)]

    for heterogeneous in [true, false]
        for sensitive in [true, false]

            if !any([heterogeneous, sensitive])
                # skip mi test
                continue
            end

            @testset "het_$heterogeneous // sens_$sensitive" begin
                sens_str = sensitive ? "fz" : "mi"
                het_str = heterogeneous ? "_nz" : ""
                test_name = sens_str * het_str
                exp_graph = exp_dict["exp_$(test_name)_maxk3"]
                pred_graph = graph(learn_network(data, sensitive=sensitive, heterogeneous=heterogeneous,
                                   max_k=3, header=header, verbose=false))

                @testset "edge_identity" begin
                    @test compare_graph_results(exp_graph, pred_graph,
                                                rtol=rtol, atol=atol,
                                                approx=true,
                                                approx_nbr_diff=approx_nbr_diff,
                                                approx_weight_meandiff=approx_weight_meandiff)
                end
            end
        end
    end

    @testset "from file" begin
        path_trunk = joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_tiny")
        for (data_format, suff_pair, transp_suff_pair) in zip(["tsv", "jld"],
                                                       [(".tsv", "_ids_transposed.tsv"),
                                                        ("_plus_meta.jld", "_plus_meta_transposed.jld")],
                                                        [("_meta.tsv", "_meta_transposed.tsv"),("","")])
            @testset "$data_format" begin
                path_pairs = [path_trunk * suff for suff in (suff_pair..., transp_suff_pair...)]
                pred_graphs = [graph(learn_network(path_pairs[i], path_pairs[i_meta], sensitive=true,
                                                   heterogeneous=false, max_k=3, verbose=false, transposed=transp))
                               for (i, i_meta, transp) in [(1, 3, false), (2, 4, true)]]

                for pred_graph in pred_graphs
                    @test compare_graph_results(pred_graphs...,
                                                rtol=rtol, atol=atol,
                                                approx=true,
                                                approx_nbr_diff=approx_nbr_diff,
                                                approx_weight_meandiff=approx_weight_meandiff)
                end
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
# out_path = joinpath("data", "learning_expected.jld2")
# save(out_path, exp_dict)
