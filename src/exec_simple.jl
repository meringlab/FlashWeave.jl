function main(input_args::Vector{String})

    println("Parsing args")
    input_path = input_args[1]
    output_path = input_args[2]
    test_name = input_args[3]
    speed_mode = input_args[4]
    parallel_mode = input_args[5]
    rec_mode = input_args[6]
    univar_mode = input_args[7]
    n_jobs = parse(Int64, input_args[8])

    println("Starting processes and importing modules")
    tic()
    if parallel_mode == "single_il"
        addprocs(1)
    elseif n_jobs > 1
        addprocs(n_jobs - 1)
    end

    eval(Expr(:using,:FlashWeave))

    println("Finished after $(toc())s\n")

    println("Reading data")
    tic()
    data_full = readdlm(input_path, '\t')
    header = convert(Array{String,1}, data_full[1, 2:end])
    #data = convert(Array{Int,2}, round(data_full[2:end, 2:end], 0))
    data = convert(Array{Float64,2}, data_full[2:end, 2:end])
    println("Finished after $(toc())s\n")

    println("Normalizing data")
    tic()
    skip_cols = Set([x for x in 1:length(header) if startswith(header[x], "ENV")])
    data_norm = Cauocc.Preprocessing.preprocess_data_default(data, test_name, verbose=false, env_cols=skip_cols)

    #if test_name == "fz_nz"
    #    zero_mask = data_norm .== minimum(data_norm)
    #    data_norm[zero_mask] = 0.0
    #end

    println("Finished after $(toc())s\n")

    #println("Clustering data")
    #tic()
    #repres, clust_dict = Cauocc.cluster_data(data_norm, test_name, parallel=split(parallel_mode, "_")[1])
    #data_norm = data_norm[:, repres]
    #println("Finished after $(toc())s\n")
    lgl_args = Dict{Symbol,Any}(:test_name => test_name, :parallel => parallel_mode, :verbose => false, :recursive_pcor => rec_mode == "true", :max_k => univar_mode == "true" ? 0 : 3)

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
    graph_dict = LGL(data_norm; lgl_args...)
    println("Finished after $(toc())s\n")

    #println("Adding cluster edges")


    println("Converting output and writing to file")
    tic()
    adj_matrix = Cauocc.Misc.dict_to_adjmat(graph_dict, header)
    writedlm(output_path, adj_matrix, '\t')
    println("Finished after $(toc())s\n")

end

main(ARGS)
