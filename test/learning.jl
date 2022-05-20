using FlashWeave
using Test, Random, StableRNGs
using SimpleWeightedGraphs
using Graphs
using SparseArrays, DelimitedFiles, Statistics, Distributed, Logging
import SimpleWeightedGraphs: nv, edges, ne, vertices, neighbors

data_path = joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_small.tsv")
data = Matrix{Float64}(readdlm(data_path, '\t')[2:end, 2:end])
data_sp = sparse(data)
header = ["X" * string(i) for i in 1:size(data, 2)]

exp_folder = joinpath("data", "learning_expected")
exp_dict = Dict(splitext(file)[1] => graph(load_network(joinpath(exp_folder, file)))
    for file in readdir(exp_folder))

rtol = 1e-2
atol = 0.0

macro silence_stdout(expr)
    quote
        orig_stdout = stdout; p1, p2 = redirect_stdout()
        res = try
            $(esc(expr))
        finally
            redirect_stdout(orig_stdout)
            close(p1), close(p2)
        end
        res
    end
end

function make_network(data, test_name, make_sparse=false, prec=64, verbose=true, return_graph=true; kwargs...)
    data_norm, mask = FlashWeave.preprocess_data_default(data, test_name, verbose=verbose,
                                                         make_sparse=make_sparse, prec=prec)
    lgl_res = FlashWeave.LGL(data_norm; test_name=test_name, verbose=verbose,  kwargs...)
    if return_graph
        lgl_res.graph
    else
        lgl_res
    end
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

T_var = 1
T_nbrs = nothing
data_bin = FlashWeave.preprocess_data_default(data, "mi", verbose=false,
                                              make_sparse=true, prec=64)[1]


nprocs() == 1 && addprocs(1)
@everywhere using FlashWeave

@testset "univar nbrs" begin
    nbrs_single = nothing
    nbrs_remote = nothing

    nbrs_dicts = []
    for (test_desc, parallel, wl) in [("smoke single", "single", true),
                                      ("smoke remote", "multi", false)]
        @testset "$test_desc" begin
            for use_pmap in [true, false]
                @testset "pmap $use_pmap" begin
                    nbrs = FlashWeave.pw_univar_neighbors(data_bin, parallel=parallel,
                                                          workers_local=wl,
                                                          levels=Int[], max_vals=Int[], cor_mat=zeros(0, 0))
                    @test isa(nbrs, Dict)
                    !use_pmap && push!(nbrs_dicts, nbrs)
                end
            end
        end
    end

    @testset "single == remote" begin
        @test nbrs_dicts[1] == nbrs_dicts[2]
    end

    global T_nbrs = nbrs_dicts[1][T_var]
end

# assure that including high-information meta variables lead to edge removal
@testset "meta conditioning" begin
    rng = StableRNG(1234)
    otu_mat_rand = rand(rng, 0:2, 100, 10)
    otu_target = rand(rng, 0:2, 100) # counts for two identical OTUs

    # meta variable identical to target OTUs, but without zeros
    mv_target = copy(otu_target)
    mv_target[mv_target .== 0] .= 1
    
    otu_mat_full = hcat(otu_mat_rand, otu_target, otu_target, mv_target)
    meta_mask = vcat(falses(12), true)

    for sensitive in [true, false]
        @testset "sensitive $sensitive" begin
            for max_k in [0, 1]
                @testset "max_k $(max_k)" begin
                    net = learn_network(otu_mat_full; sensitive=sensitive, heterogeneous=true, max_k=max_k, verbose=false, 
                        meta_mask=meta_mask, normalize=false)
                    num_edges = SimpleWeightedGraphs.ne(graph(net))
                    if max_k == 0
                        @test num_edges == 3
                    else # with conditioning, one of the three edges is explained away (the others stay due to heuristic)
                        @test num_edges == 2
                    end
                end
            end
        end
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


                                    pred_graph = @silence_stdout begin
                                                    make_network(data, test_name, make_sparse, 64,
                                                                 max_k=max_k, parallel=parallel, time_limit=time_limit)
                                                 end

                                    # special case for conditional mi
                                    if test_name == "mi" && max_k == 3
                                        approx_nbr_diff = 22
                                        approx_weight_meandiff = 0.16
                                    else
                                        approx_nbr_diff = 0
                                        approx_weight_meandiff = 0.05
                                    end

                                    @testset "edge_identity" begin
                                        @test compare_graph_results(exp_graph, pred_graph,
                                                                    rtol=rtol, atol=atol,
                                                                    approx=true,
                                                                    approx_nbr_diff=approx_nbr_diff,
                                                                    approx_weight_meandiff=approx_weight_meandiff)
                                    end

                                    @testset "nonzero_weights" begin
                                        @test !any(pred_graph.weights.nzval .== 0.0)
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

@testset "pcor_recursive_fits_iterative" begin
    approx_nbr_diff = 0
    approx_weight_meandiff = 0.05
    
    for test_name in ["fz", "fz_nz"]
        @testset "$test_name" begin
            for prec in [32, 64]
                @testset "precision $prec" begin
                    pred_graph_iter = make_network(data, test_name, false, prec, false; recursive_pcor=false)
                    pred_graph_rec = make_network(data, test_name, false, prec, false)
                    @testset "edge_identity" begin
                        @test compare_graph_results(pred_graph_iter, pred_graph_rec,
                                                    rtol=rtol, atol=atol,
                                                    approx=true,
                                                    approx_nbr_diff=approx_nbr_diff,
                                                    approx_weight_meandiff=approx_weight_meandiff)
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
                pred_graph = @silence_stdout make_network(data, test_name, make_sparse, 32, max_k=3, parallel="single_il", time_limit=0.0)
                exp_graph = exp_dict["exp_$(test_name)_maxk3"]
                @test compare_graph_results(exp_graph, pred_graph, rtol=1e-2, atol=0.0)
            end
        end
    end
end


@testset "learn_network" begin
    approx_nbr_diff = 0
    approx_weight_meandiff = 0.05


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

                pred_netw = @silence_stdout begin
                                learn_network(data, sensitive=sensitive, heterogeneous=heterogeneous,
                                            max_k=3, header=header, track_rejections=true, verbose=true)
                            end
                pred_graph = graph(pred_netw)

                @testset "edge_identity" begin
                    @test compare_graph_results(exp_graph, pred_graph,
                                                approx=true,
                                                approx_nbr_diff=approx_nbr_diff,
                                                approx_weight_meandiff=approx_weight_meandiff)
                end

                @testset "show" begin
                    @test @silence_stdout isa(show(pred_netw), Nothing)
                end
            end
        end
    end

    @testset "from file" begin
        path_trunk = joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_tiny")
        for (data_format, suff_pair, transp_suff_pair) in zip(["tsv", "jld2"],
                                                        [(".tsv", "_ids_transposed.tsv"),
                                                        ("_plus_meta.jld2", "_plus_meta_transposed.jld2")],
                                                        [("_meta.tsv", "_meta_transposed.tsv"),("","")])
            data_format == "jld2" && continue
            @testset "$data_format" begin
                path_pairs = [path_trunk * suff for suff in (suff_pair..., transp_suff_pair...)]

                # skip jld2


                pred_graphs = []
                for (i, i_meta, transp) in [(1, 3, false), (2, 4, true)]
                    pred_netw = learn_network(path_pairs[i], path_pairs[i_meta],
                                    sensitive=true, heterogeneous=false,
                                    max_k=3, verbose=false, transposed=transp,
                                    n_obs_min=0)
                    push!(pred_graphs, graph(pred_netw))
                end

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

    @testset "one hot" begin
        path_trunk = joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_tiny")
        oh_paths = [path_trunk * suff for suff in (".tsv", "_meta_oneHotTest.tsv")]
        for sensitive in (true, false)
            for heterogeneous in (true, false)
                @testset "het_$heterogeneous // sens_$sensitive" begin
                    pred_graph = @silence_stdout begin
                                    graph(learn_network(oh_paths..., sensitive=sensitive,
                                                heterogeneous=heterogeneous, max_k=3,
                                                verbose=true, transposed=false, n_obs_min=0))
                                end
                    @test @silence_stdout isa(show(pred_graph), Nothing)
                end
            end
        end
    end
end

# smoke test fast elimination heuristic
@testset "fast_elim" begin
    @test isa(learn_network(data, sensitive=true, heterogeneous=false,
                                         max_k=3, header=header, fast_elim=false, verbose=false), FlashWeave.FWResult)
end

@testset "duplicates" begin
    dupl_path = joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_small_1to5duplic_noIDs.tsv")
    data_dupl = readdlm(dupl_path, Float64)
    pred_graph = @silence_stdout graph(learn_network(data_dupl))
    @testset "nonzero_weights" begin
        @test !any(pred_graph.weights.nzval .== 0.0)
    end
end

@testset "convergence" begin
    @test @silence_stdout begin
                isa(show(FlashWeave.FWResult(make_network(data, "fz", true, 64, true, false,
                        convergence_threshold=Inf, max_k=3, parallel="single_il",
                        time_limit=1e-8, update_interval=0.001))), Nothing)
          end
end

@testset "hiton msg" begin
    @test @silence_stdout begin
                isa(FlashWeave.si_HITON_PC(T_var, data_bin, Int[], Int[], zeros(0, 0),
                              univar_nbrs=T_nbrs, debug=2), FlashWeave.HitonState)
          end
end

# smoke test bnb heuristic
@testset "bnb heuristic" begin
    for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
        for make_sparse in [true, false]
            for cb in [true, false]
                @testset "$(test_name)_$(make_sparse)_cb$(cb)_single" begin
                    pred_graph = @silence_stdout begin
                        make_network(data, test_name, make_sparse, 64, max_k=3,
                        parallel="single", time_limit=0.0; bnb=true, cut_test_branches=cb)
                    end
                    @test isa(pred_graph, SimpleWeightedGraph)
                end
            end
        end
    end
end

# assure that tables with variables that are observed everywhere are handled correctly
@testset "non-zero variables" begin
    rng = StableRNG(1234)
    A = rand(rng, 1:1000, 100, 10)
    A[rand(rng, Bool, 100, 10)] .= 0
    A[:, end] .+= 1 # enforce one variable without zeros
    for sensitive in [true, false]
        @testset "sensitive $sensitive" begin
            for heterogeneous in [true, false]
                @testset "heterogeneous $heterogeneous" begin
                    for max_k in [0, 1]
                        @testset "max_k $(max_k)" begin
                            @test isa(learn_network(A; sensitive=sensitive, heterogeneous=heterogeneous, max_k=max_k, verbose=false, normalize=true), FlashWeave.FWResult)
                        end
                    end
                end
            end
        end
    end
end

@testset "shared data" begin
    approx_nbr_diff = 0
    approx_weight_meandiff = 0.05

    for make_sparse in [true, false]
        @testset "sparse_$(make_sparse)" begin
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

                        pred_netw = @silence_stdout begin
                                        learn_network(data, sensitive=sensitive, heterogeneous=heterogeneous,
                                                    max_k=3, header=header, track_rejections=true, verbose=true, share_data=true)
                                    end
                        pred_graph = graph(pred_netw)

                        @testset "edge_identity" begin
                            @test compare_graph_results(exp_graph, pred_graph,
                                                        approx=true,
                                                        approx_nbr_diff=approx_nbr_diff,
                                                        approx_weight_meandiff=approx_weight_meandiff)
                        end
                    end
                end
            end
        end
    end
end


# to create expected output

#for test_name in ["fz", "fz_nz", "mi", "mi_nz"]
#    for max_k in [0, 3]
#        sensitive = startswith(test_name, "fz")
#        heterogeneous = endswith(test_name, "_nz")
#        net = learn_network(data, sensitive=sensitive, heterogeneous=heterogeneous, prec=64, max_k=max_k,
#            parallel_mode="single_il", time_limit=30.0, verbose=false)
#        path = joinpath("data", "learning_expected", "exp_$(test_name)_maxk$(max_k).edgelist")
#        save_network(path, net)
#     end
#end

