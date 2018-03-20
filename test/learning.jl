using FlashWeave
using JLD2, FileIO
using Base.Test
using MetaGraphs
using LightGraphs

#cd("/Users/janko/.julia/v0.6/FlashWeave/test")
#data = Array(readtable(joinpath("test", "data", "HMP_SRA_gut_small.tsv"))[:, 2:end])
data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])

exp_dict = load(joinpath("data", "learning_expected.jld"))

function make_network(data, test_name, make_sparse=false, prec=32, verbose=false; kwargs...)
    data_norm = FlashWeave.Preprocessing.preprocess_data_default(data, test_name, verbose=false, make_sparse=make_sparse, prec=prec)
    kwargs_dict = Dict(kwargs)
    graph_res = LGL(data_norm; test_name=test_name, verbose=verbose, weight_type="cond_logpval",  kwargs...)
    graph_res.graph
end

function compare_graph_results(g1::Dict, g2::MetaGraph; verbose=false, rtol=0.0, atol=0.0)
    if Set(keys(g1)) != Set(vertices(g2))
        if verbose
            println("Upper level keys don't match")
        end
        return false
    end

    for T in keys(g1)
        nbr_dict1 = g1[T]

        if Set(keys(nbr_dict1)) != Set(neighbors(g2, T))
            if verbose
                println("Neighbors for node $T dont match")
            end
            return false
        end

        for nbr in keys(nbr_dict1)
            g2_weight = get_prop(g2, T, nbr, :weight)
            if !isapprox(nbr_dict1[nbr], g2_weight, rtol=rtol, atol=atol)
                if verbose
                    println("Weights for node $T and neighbor $nbr dont fit: $(nbr_dict1[nbr]), $(g2_weight)")
                end
                return false
            end
        end
    end
    true
end

# For sanity checking
#max_k = 0
#make_sparse = false
#parallel = "single"
#test_name = "mi"
#graph = make_network(data, test_name, make_sparse, 64, true, max_k=max_k, parallel=parallel, time_limit=30.0, correct_reliable_only=false, n_obs_min=0, debug=0, verbose=true, FDR=false)


#wanted_vars = Set([1,2,3])
#graph2 = make_network(data, test_name, make_sparse, 64, true, max_k=max_k, parallel=parallel, time_limit=30.0, correct_reliable_only=false, n_obs_min=0, debug=0, verbose=true, wanted_vars=wanted_vars, FDR=false)

#wanted_vars2 = Set([1,25,50])
#graph3 = make_network(data, test_name, make_sparse, 64, true, max_k=max_k, parallel=parallel, time_limit=30.0, correct_reliable_only=false, n_obs_min=0, debug=0, verbose=true, wanted_vars=wanted_vars2, FDR=false)

#[(x, Set(neighbors(graph, x)) == Set(neighbors(graph2, x))) for x in wanted_vars]
#[(x, Set(neighbors(graph, x)) == Set(neighbors(graph3, x))) for x in wanted_vars2]

#@code_warntype make_network(data, test_name, make_sparse, 64, max_k=max_k, parallel=parallel, time_limit=30.0, correct_reliable_only=false, n_obs_min=0, verbose=true)

#exp_graph_dict = exp_dict["exp_$(test_name)_maxk$(max_k)_paramulti_il"]
#atol = 1e-2
#rtol = 0.0
#println(compare_graph_results(exp_graph_dict, graph, rtol=rtol, atol=atol, verbose=true))
#println(keys(exp_dict))

@testset "major_test_modes" begin
    for test_name in ["mi", "mi_nz", "fz", "fz_nz"]#(test_name, sub_dict) in exp_num_nbr_dict
        @testset "$test_name" begin
            for max_k in [0, 3]#(max_k, exp_num_nbr) in sub_dict
                @testset "max_k $max_k" begin
                    for make_sparse in [true, false]
                        @testset "sparse $make_sparse" begin
                            for parallel in ["single", "multi_il"]#["single", "multi_il"]
                                @testset "parallel $parallel" begin
                                    time_limit = endswith(parallel, "il") ? 30.0 : 0.0
                                    graph = make_network(data, test_name, make_sparse, 64, max_k=max_k, parallel=parallel,
                                                         time_limit=time_limit, correct_reliable_only=false, n_obs_min=0)

                                    if parallel != "single_il"
                                        exp_graph_dict = exp_dict["exp_$(test_name)_maxk$(max_k)_para$(parallel)"]

                                        atol = 1e-2
                                        rtol = 0.0

                                        compare_graph_results(exp_graph_dict, graph, rtol=rtol, atol=atol)


                                        @testset "edge_identity" begin
                                            @test compare_graph_results(exp_graph_dict, graph, rtol=rtol, atol=atol)
                                        end
                                    else
                                        @testset "has_edges" begin
                                            @test ne(graph) > 0
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
end

@testset "precision_32" begin
    for (test_name, make_sparse) in [("mi_nz", true), ("fz", false)]
        for make_sparse in [true]
            @testset "$(test_name)_$(make_sparse)_single" begin
                graph = make_network(data, test_name, make_sparse, 32, max_k=3, parallel="single", time_limit=0.0,
                    correct_reliable_only=false, n_obs_min=0)
                exp_graph_dict = exp_dict["exp_$(test_name)_maxk3_parasingle"]
                @test compare_graph_results(exp_graph_dict, graph, rtol=0.0, atol=1e-2)
            end
        end
    end

    #@testset "fz_nz_nonsparse_multi_il" begin
    #    graph_dict = make_network(data, "fz_nz", false, 32, max_k=3, parallel="multi_il", time_limit=0.0,
    #        correct_reliable_only=false)
    #    exp_graph_dict = exp_dict["exp_fz_nz_maxk3_paramulti_il"]
    #    @test compare_graph_dicts(graph_dict, exp_graph_dict, rtol=0.0, atol=1e-2)
    #end
end


@testset "no_red_tests_OFF" begin
    for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
        graph = make_network(data, test_name, false, 64, max_k=3, parallel="single", time_limit=0.0, no_red_tests=false,
                                  correct_reliable_only=false, n_obs_min=0)
        exp_graph_dict = exp_dict["exp_$(test_name)_maxk3_parasingle"]
        atol = 1e-2
        rtol = 0.0

        @testset "$test_name" begin
            @test compare_graph_results(exp_graph_dict, graph, rtol=rtol, atol=atol)
        end
    end
end

#@testset "wanted_vars" begin
#    for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
#        @testset "$test_name" begin
#        end
#    end
#end

#@testset "fast_elim_OFF" begin
#    for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
#        graph_dict = make_network(data, test_name, false, 64, max_k=3, parallel="single", time_limit=0.0, fast_elim=false)
#        exp_graph_dict = exp_dict["exp_$(test_name)_maxk3_parasingle"]
#        atol = 1e-2
#        rtol = 0.0
#
#        @testset "$test_name" begin
#            @test compare_graph_dicts(graph_dict, exp_graph_dict, rtol=rtol, atol=atol)
#        end
#    end
#end


#@testset "track_rejections" begin
#    exp_num_nbr = exp_num_nbr_dict["fz"][3]
#    @test all(get_num_nbr(data, "fz", false, max_k=3, parallel="single", track_rejections=true) .== exp_num_nbr)
#    exp_num_nbr = exp_num_nbr_dict["mi"][0]
#    @test all(get_num_nbr(data, "mi", false, max_k=0, parallel="single", track_rejections=true) .== exp_num_nbr)
#end

#@testset "speed" begin
#    for test_name in ["mi", "mi_nz", "fz", "fz_nz"]
#        println(test_name)
#
#        for max_k in [0, 3]
#            println("\t$max_k")
#
#            @testset "$test_name $max_k" begin
#                test_times = []
#                for make_sparse in [true, false]
#                    for prec in [32, 64]
#                        println("\t\t$make_sparse $prec")
#                        data_norm = FlashWeave.Preprocessing.preprocess_data_default(data, test_name, verbose=false, make_sparse=make_sparse, prec=prec)
#                        start_time = time()
#                        LGL(data_norm; test_name=test_name, verbose=false, max_k=max_k)
#                        time_taken = time() - start_time
#                        push!(test_times, time_taken)
#                    end
#                end
#            end
#        end
#    end
#end

#@testset "preclustering" begin
#    test_name = "fz"
#    make_sparse = false
#    max_k = 3
#    precluster_sim = 0.2
#    @testset "representatives" begin
#        exp_num_nbr = [1,1,0,2,0,0,1,0,0,2,0,1,2,1,0,0,0,0,
#                       2,1,0,0,0,0,2,0,1,0,1,0,1,1]
#        @test_broken all(get_num_nbr(data, test_name, make_sparse, max_k=max_k, parallel="single", precluster_sim=precluster_sim, fully_connect_clusters=false) .== exp_num_nbr)
#    end
#    @testset "fully_connect__track_rej" begin
#        exp_num_nbr = [1,2,1,1,1,1,5,0,2,0,2,1,1,3,0,3,0,5,3,1,
#                       4,2,1,4,1,0,1,1,3,2,4,1,2,0,2,1,3,0,3,0,
#                       2,1,0,2,0,1,1,0,2,2]
#        @test_broken all(get_num_nbr(data, test_name, make_sparse, max_k=max_k, parallel="single", precluster_sim=precluster_sim, fully_connect_clusters=true, track_rejections=true) .== exp_num_nbr)
#    end
#end

# to create expected output

 # exp_dict = Dict()
 # for (test_name, sub_dict) in exp_num_nbr_dict
 #     for (max_k, exp_num_nbr) in sub_dict
 #         for make_sparse in [true, false]
 #             for parallel in ["single", "multi_il"]
 #                 graph_dict = make_network(data, test_name, make_sparse, max_k=max_k, parallel=parallel, time_limit=0.0)
 #                  exp_dict["exp_$(test_name)_maxk$(max_k)_para$(parallel)_sparse$(make_sparse)"] = graph_dict
 #              end
 #          end
 #      end
 #  end
 #
 #  out_path = joinpath(pwd(), "test", "data", "learning_expected.jld")
 #  rm(out_path)
 #  save(out_path, exp_dict)
