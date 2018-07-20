# note: needed lots of @eval and Base.invokelatest hacks for conditional
# module loading

const valid_net_formats = (".edgelist", ".jld2")
const valid_data_formats = (".tsv", ".csv", ".biom", ".jld2")

function load_data(data_path::AbstractString, meta_path=nothing; transposed::Bool=false,
     data_key::AbstractString="data", meta_key::AbstractString="meta_data",
     header_key::AbstractString="header", meta_header_key::AbstractString="meta_header")
     """Load OTU tables and meta data from various formats.
     -- Set jld keys you don't want to use to 'nothing'
     -- delimited formats must have headers (or row indices if transposed=true)"""

    file_ext = splitext(data_path)[2]
    if file_ext in [".tsv", ".csv"]
        ld_results = load_dlm(data_path, meta_path, transposed=transposed)
    elseif file_ext == ".biom"
        ld_results = load_biom(data_path, meta_path)
    elseif file_ext == ".jld2"
        ld_results = load_jld(data_path, data_key, header_key, meta_key, meta_header_key)
    else
        error("$(file_ext) not a valid output format. Choose one of $(valid_data_formats)")
    end

    ld_results
end

function save_network(out_path::AbstractString, net_result::LGLResult; detailed::Bool=false)
    file_ext = splitext(out_path)[2]
    if file_ext == ".edgelist"
        write_edgelist(out_path, net_result.graph)
    elseif file_ext == ".jld2"
        isdefined(:FileIO) || @eval using FileIO: save, load
        Base.invokelatest(save, out_path, "results", net_result)
    else
        error("$(file_ext) not a valid output format. Choose one of $(valid_net_formats)")
    end

    if detailed
        out_trunk = splitext(out_path)[1]
        save_rejections(out_trunk * "_rejections.tsv", net_result)
        save_unfinished_variable_info(out_trunk * "_unchecked.tsv", net_result)
    end
end

function load_network(net_path::AbstractString)
    file_ext = splitext(net_path)[2]
    if file_ext == ".edgelist"
        G = read_edgelist(net_path)
        net_result = LGLResult(G)
    elseif file_ext == ".jld2"
        isdefined(:FileIO) || @eval using FileIO: save, load
        d = Base.invokelatest(load, net_path)
        net_result = d["results"]
    else
        error("$(file_ext) not a valid network format. Valid formats are $(valid_net_formats)")
    end
end


######################
## Helper functions ##
######################

function load_jld(data_path::AbstractString, data_key::AbstractString, header_key::AbstractString,
     meta_key=nothing, meta_header_key=nothing)
     isdefined(:FileIO) || @eval using FileIO: save, load
     d = Base.invokelatest(load, data_path)

     data = d[data_key]
     header = d[header_key]

     if meta_key != nothing
         meta_data = d[meta_key]
         meta_header = d[meta_header_key]
     else
         meta_data = meta_header = nothing
     end

     data, header, meta_data, meta_header
 end


hasrowids(data::AbstractMatrix) = data[1, 1] == "" || isa(data[2, 1], AbstractString)

function load_dlm(data_path::AbstractString, meta_path=nothing; transposed::Bool=false)
    sep = splitext(data_path)[2] == ".tsv" ? '\t' : ','

    data = readdlm(data_path, sep)

    if transposed
        data = data'
    end

    if hasrowids(data)
        data = data[:, 2:end]
    end

    header = Vector{String}(data[1, :])[:]
    data = Matrix{Float64}(data[2:end, :])

    if meta_path != nothing
        meta_data, meta_header = load_dlm(meta_path, transposed=transposed)
    else
        meta_data = meta_header = nothing
    end

    data, header, meta_data, meta_header
end



function load_biom_json(data_path)
    json_struc = JSON.parsefile(data_path)
    otu_table = Matrix{Int}(hcat(json_struc["data"]...))

    if json_struc["matrix_type"] == "sparse"
        otu_table = otu_table'
        otu_table = sparse(otu_table[:, 1] + 1, otu_table[:, 2] + 1, otu_table[:, 3])'
    end

    header = [x["id"] for x in json_struc["rows"]]
    otu_table, header
end


function load_biom_hdf5(data_path)
    f = h5open(data_path, "r")
    m, n = read(attrs(f)["shape"])
    colptr, rowval, nzval = [read(f, "sample/matrix/$key") for key in ["indptr", "indices", "data"]]
    otu_table = SparseMatrixCSC(m, n, colptr + 1, rowval + 1, Vector{Int}(nzval))'
    header = read(f, "observation/ids")
    close(f)

    otu_table, header
end


function load_biom(data_path, meta_path=nothing)
    data, header = try
        isdefined(:HDF5) || @eval using HDF5
        Base.invokelatest(load_biom_hdf5, data_path)
    catch
        try
            load_biom_json(data_path)
        catch
            error("file $data_path is not valid .biom")
        end
    end

    if meta_path != nothing
        meta_data, meta_header = load_dlm(meta_path)
    else
        meta_data = meta_header = nothing
    end

    data, header, meta_data, meta_header
end


function save_rejections(rej_path, net_result)
    rej_dict = net_result.rejections

    open(rej_path, "w") do f
        if isempty(rej_dict)
            write(f, "# No rejections found, you may have forgotten to specify 'track_rejections' when running FlashWeave")
        else
            write(f, join(["Edge", "Rejecting_set", "Stat", "P_value", "Num_tests", "Perc_tested"], "\t"), "\n")
            for (var_A, rej_nbr_dict) in rej_dict
                for (var_B, rej_info) in rej_nbr_dict
                    test_res = rej_info[2]
                    stat, pval = test_res.stat, test_res.pval
                    rej_set = rej_info[1]
                    num_tests, frac_tested = rej_info[3]

                    line_items = [string(var_A) * " <-> " * string(var_B), join(rej_set, ","), round(stat, 5), pval, num_tests, round(frac_tested, 3)]
                    write(f, join(line_items, "\t"), "\n")
                end
            end
        end
    end
end


function save_unfinished_variable_info(unf_path, net_result)
    unf_states_dict = net_result.unfinished_states
    open(unf_path, "w") do f
        if isempty(unf_states_dict)
            write(f, "# No unchecked neighbors")
        else
            write(f, "Variable", "\t", "Phase", "\t", "Unchecked_neighbors", "\n")
            for (var_A, unf_state) in unf_states_dict
                unf_nbrs = unf_state.unchecked_vars
                phase = unf_state.phase
                write(f, string(var_A), "\t", phase, "\t", join(unf_nbrs, ","), "\n")
            end
        end
    end
end



function write_edgelist(out_path::AbstractString, G::SimpleWeightedGraph; header=nothing)
    open(out_path, "w") do out_f
        for e in edges(G)
            if header == nothing
                e1 = e.src
                e2 = e.dst
            else
                e1 = header[e.src]
                e2 = header[e.dst]
            end
            write(out_f, string(e1) * "\t" * string(e2) * "\t" * string(G.weights[e.src, e.dst]), "\n")
        end
    end
end


function read_edgelist(in_path::AbstractString; header=nothing)
    srcs = Int[]
    dsts = Int[]
    ws = Float64[]

    if header != nothing
        inv_header = Dict{eltype(header), Int}(zip(header, 1:length(header)))
    end

    open(in_path, "r") do in_f
        for line in eachline(in_f)
            line_items = split(chomp(line), '\t')

            if header != nothing
                src = inv_header[line_items[1]]
                dst = inv_header[line_items[2]]
            else
                src = parse(Int, line_items[1])
                dst = parse(Int, line_items[2])
            end

            push!(srcs, src)
            push!(dsts, dst)
            push!(ws, parse(Float64, line_items[end]))
        end
    end
    SimpleWeightedGraph(srcs, dsts, ws)
end
