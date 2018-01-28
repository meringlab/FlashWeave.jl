println("Starting processes and importing modules")
tic()
#if endswith(ARGS[5], "_il")
#    #addprocs(1)
#end

using FlashWeave
using JLD2
toc()

function main(input_args::Vector{String})

    println("Parsing args")
    input_path = input_args[1]
    output_path = input_args[2]
    test_name = input_args[3]
    speed_mode = input_args[4]
    parallel_mode = input_args[5]
    rec_mode = input_args[6]
    univar_mode = input_args[7]
    FDR = input_args[8]
    make_sparse = input_args[9]
    normalize = input_args[10]
    write_table = input_args[11]
    max_k = input_args[12]
    weight_type = input_args[13]

    #println("Starting processes and importing modules")
    #tic()
    #if parallel_mode == "single_il"
    #    addprocs(1)
    #elseif n_jobs > 1
    #    addprocs(n_jobs - 1)
    #end

    #eval(Expr(:using,:FlashWeave))

    #println("Finished after $(toc())s\n")

    println("Reading data")
    tic()
    if endswith(input_path, ".jld")
        data_dict = load(input_path)
        data = data_dict["data"]
        header = data_dict["header"]
    else
        data_full = readdlm(input_path, '\t')
        header = convert(Array{String,1}, data_full[1, 2:end])
        #data = convert(Matrix{Int}, round.(data_full[2:end, 2:end], 0))
        data = convert(Matrix{Float64}, data_full[2:end, 2:end])
        #println(typeof(data))
        #println(test_name)
        #println(typeof(data) <: AbstractMatrix{Real})
        #println(data)
        #println("Finished after $(toc())s\n")
    end
    toc()
    
    if normalize == "true"
        println("Normalizing data")
        tic()

        skip_cols = Int[x for x in 1:length(header) if startswith(header[x], "ENV")]

        data_norm, header = FlashWeave.Preprocessing.preprocess_data_default(data, test_name, verbose=false, env_cols=skip_cols, header=header, make_sparse=make_sparse == "true")
        toc()
    else
        data_norm = data
    end
    #data_norm = FlashWeave.Preprocessing.preprocess_data_default(data, test_name, verbose=true, env_cols=skip)
    #if test_name == "fz_nz"
    #    zero_mask = data_norm .== minimum(data_norm)
    #    data_norm[zero_mask] = 0.0
    #end

    #println("Finished after $(toc())s\n")
    #toc()
    #println("Clustering data")
    #tic()
    #repres, clust_dict = Cauocc.cluster_data(data_norm, test_name, parallel=split(parallel_mode, "_")[1])
    #data_norm = data_norm[:, repres]
    #println("Finished after $(toc())s\n")
    if univar_mode == "true"
        max_k = 0
    else
        max_k = parse(Int, max_k)
    end
    
    lgl_args = Dict{Symbol,Any}(:test_name => test_name, :parallel => parallel_mode, :verbose => false, :recursive_pcor => rec_mode == "true", :max_k => max_k, :FDR => FDR == "true", :weight_type => weight_type)

    if speed_mode == "fast"
        lgl_args[:convergence_threshold] = 0.05
        lgl_args[:fast_elim] = true
    elseif speed_mode == "precise"
        lgl_args[:convergence_threshold] = 0.0
        lgl_args[:fast_elim] = false
    else
        error("$speed_mode not a valid speed_mode")
    end

    println("Learning network")
    tic()
    graph = LGL(data_norm; lgl_args...).graph
    #println("Finished after $(toc())s\n")
    toc()
    #println("Adding cluster edges")

    if write_table == "true"
        println("Converting output and writing to file")
        tic()
        #adj_matrix = FlashWeave.Misc.dict_to_adjmat(graph, header)
        adj_matrix = FlashWeave.Misc.metagraph_to_adjmat(graph, header)
        writedlm(output_path, adj_matrix, '\t')
        #println("Finished after $(toc())s\n")
        toc()
    end
end

main(ARGS)
