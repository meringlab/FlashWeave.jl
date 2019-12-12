using Test
using FlashWeave
using SimpleWeightedGraphs, DataStructures
using SparseArrays, DelimitedFiles, FileIO

#net_result = load_network(joinpath("data", "io_expected_networks.jld2"))
net_result = load_network(joinpath("data", "learning_expected", "exp_mi_maxk3.edgelist"))

function compare_rejections(rej1, rej2; approx_kwargs...)
    for (item1, item2) in zip(rej1, rej2)
        cmp = if isa(item1, Tuple)
            all(isapprox.(item1, item2; approx_kwargs...))
        elseif isa(item1, FlashWeave.TestResult)
            all(isapprox(getproperty(item1, p), getproperty(item2, p); approx_kwargs...) for p in propertynames(item1))
        else
            isapprox(item1, item2; approx_kwargs...)
        end
        !cmp && return false
    end
    true
end

function make_reduced_HitonState(phase, unchecked_vars)
    dummy_d = OrderedDict{Int,Tuple{Float64,Float64}}()
    dummy_rej = FlashWeave.RejDict{Int}()
    FlashWeave.HitonState(phase, dummy_d, dummy_d, unchecked_vars, dummy_rej)
end

@testset "networks" begin
    tmp_path = tempname()

    for net_format in ["edgelist", "gml", "jld2"]

        # skip jld2
        net_format == "jld2" && continue

        @testset "$net_format" begin
            tmp_net_path = tmp_path * "." * net_format
            save_network(tmp_net_path, net_result)
            net_result_ld = load_network(tmp_net_path)
            @test graph(net_result_ld) == graph(net_result)
        end
    end

    @testset "detailed" begin
        tmp_net_path = tmp_path * ".edgelist"
        rej_dict = Dict(1=>Dict(2=>((3,), FlashWeave.TestResult(0.0, 1.0, 1, false), (10, 0.2)),
                                3=>((4,5,6), FlashWeave.TestResult(0.1, 0.2, 3, true), (1000, 1.0))))
        unf_dict = Dict(1=>make_reduced_HitonState('I', [7, 8, 9]),
                        2=>make_reduced_HitonState('E', [11]))

        det_lgl_res = FlashWeave.LGLResult(graph(net_result), rej_dict, unf_dict)
        det_net_result = FlashWeave.FWResult(det_lgl_res, variable_ids=net_result.variable_ids,
                                             meta_variable_mask=net_result.meta_variable_mask)
        save_network(tmp_net_path, det_net_result, detailed=true)

        @testset "rejections" begin
            rej_dict_ld = FlashWeave.load_rejections(tmp_path * "_rejections.tsv")
            @test all(compare_rejections(rej_dict_ld[k][v], rej_dict[k][v]) for k in keys(rej_dict_ld) for v in keys(rej_dict_ld[k]))
        end

        @testset "unfinished states" begin
            println(tmp_path * "_unchecked.tsv")
            unf_dict_ld = FlashWeave.load_unfinished_variable_info(tmp_path * "_unchecked.tsv")
            @test begin
                all(all(getproperty(unf_dict_ld[i], p) == getproperty(unf_dict[i], p)
                for p in (:phase, :unchecked_vars) for i in keys(unf_dict_ld)))
            end
        end
    end
end


data, header = readdlm(joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_small.tsv"), '\t', header=true)
data = Matrix{Int}(data[1:19, 2:20])
header = Vector{String}(header[2:20])
meta_data, meta_header = readdlm(joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_tiny_meta.tsv"), '\t', Int, header=true)
meta_header = Vector{String}(meta_header[:])
meta_data_key = "meta_data"
meta_header_key = "meta_header"
meta_data_fact, meta_header_fact = readdlm(joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_tiny_meta_oneHotTest.tsv"), '\t', header=true)
meta_header_fact = meta_header_fact[:]


@testset "table data" begin
    for (data_format, data_suff, meta_suff) in zip(["tsv", "tsv_rownames", "csv", "biom_json", "biom_hdf5", "jld2"],
                                                   [".tsv", "_ids.tsv", ".csv", "_json.biom", "_hdf5.biom", "_plus_meta.jld2"],
                                                   ["_meta.tsv", "_meta.csv", "_meta.csv", "_meta.tsv", "_meta.tsv", ""])
        # skip jld2
        data_format == "jld2" && continue

        @testset "$data_format" begin
            data_path, meta_path = [joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_tiny" * suff) for suff in [data_suff, meta_suff]]
            data_ld = load_data(data_path, meta_path, meta_data_key=meta_data_key, meta_header_key=meta_header_key)
            @test data_ld[1] == data
            @test data_ld[2] == header
            @test data_ld[3] == meta_data
            @test data_ld[4] == meta_header
        end
    end
end


@testset "transposed" begin
    tmp_path = tempname()

    for (data_format, data_suff, meta_suff) in zip(["tsv", "jld2"],
                                                   ["_ids_transposed.tsv", "_plus_meta_transposed.jld2"],
                                                   ["_meta_transposed.tsv", ""])
        @testset "$data_format" begin
            data_path, meta_path = [joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_tiny" * suff) for suff in [data_suff, meta_suff]]
            data_ld = load_data(data_path, meta_path, transposed=true, meta_data_key=meta_data_key,
                                meta_header_key=meta_header_key)
            @test data_ld[1] == data
            @test data_ld[2] == header
            @test data_ld[3] == meta_data
            @test data_ld[4] == meta_header
        end
    end
end


@testset "string factors" begin
    path_prefix = joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_tiny")
    data_path, meta_path = [path_prefix * suff for suff in ("_ids.tsv", "_meta_oneHotTest.tsv")]
    data_ld = load_data(data_path, meta_path, meta_data_key=meta_data_key,
                        meta_header_key=meta_header_key)
    @test data_ld[1] == data
    @test data_ld[2] == header
    @test data_ld[3] == meta_data_fact
    @test data_ld[4] == meta_header_fact
end

@testset "numeric IDs" begin
    data_path = joinpath("data", "HMP_SRA_gut", "HMP_SRA_gut_tiny_numIDs.tsv")
    data_ld = load_data(data_path)
    @test data_ld[1] == data
    @test data_ld[2] == map(x -> x[3:end], header)
end





# to create expected output
# using FileIO
# using FlashWeave
#
# function make_network(data, test_name, make_sparse=false, prec=64, verbose=false; kwargs...)
#     data_norm = FlashWeave.preprocess_data_default(data, test_name, verbose=false, make_sparse=make_sparse, prec=prec)
#     kwargs_dict = Dict(kwargs)
#     graph_res = FlashWeave.LGL(data_norm; test_name=test_name, verbose=verbose,  kwargs...)
#     graph_res
# end
#
# data = Matrix{Float64}(readdlm(joinpath("data", "HMP_SRA_gut_small.tsv"), '\t')[2:end, 2:end])
#
# max_k = 3
# make_sparse = false
# parallel = "single"
# test_name = "mi"
# lgl_res = make_network(data, test_name, make_sparse, 64, true, max_k=max_k, parallel=parallel, time_limit=30.0, correct_reliable_only=false, n_obs_min=0, debug=0, verbose=true, FDR=true, weight_type="cond_stat")
# save(joinpath("data", "io_expected.jld2"), "results", lgl_res)
