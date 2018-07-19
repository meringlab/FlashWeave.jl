module Io

using LightGraphs, SimpleWeightedGraphs
using JLD2, FileIO
using JSON, HDF5

using FlashWeave.Types

export load_data, save_network, load_network

const valid_net_formats = (".edgelist", ".jld2")
const valid_data_formats = (".tsv", ".csv", ".biom")

function load_data(data_path::AbstractString, meta_path=nothing; data_key="data",
     meta_key="meta_data", header_key="header", meta_header_key="meta_header")
     """Load OTU tables and meta data from various formats.
     -- Set jld keys you don't want to use to 'nothing'"""

    file_ext = splitext(data_path)[2]
    if file_ext in [".tsv", ".csv"]
        ld_results = load_dlm(data_path, meta_path)
    elseif file_ext == ".biom"
        ld_results = load_biom(data_path, meta_path)
    elseif file_ext == ".jld2"
        ld_results = load_jld(data_path, data_key, header_key, meta_key, meta_header_key)
    else
        error("$(file_ext) not a valid output format. Choose one of $(valid_data_formats)")
    end

    ld_results
end

function save_network(out_path::AbstractString, net_result::LGLResult; detailed=false)
    file_ext = splitext(out_path)[2]
    if file_ext == ".edgelist"
        write_edgelist(out_path, net_result.graph)
    elseif file_ext == ".jld2"
        save(out_path, "results", net_result)
    else
        error("$(file_ext) not a valid output format. Choose one of $(valid_net_formats)")
    end
end

function load_network(net_path::AbstractString)
    file_ext = splitext(net_path)[2]
    if file_ext == ".edgelist"
        G = read_edgelist(net_path)
        net_result = LGLResult(G)
    elseif file_ext == ".jld2"
        net_result = load(net_path)["results"]
    else
        error("$(file_ext) not a valid network format. Valid formats are $(valid_net_formats)")
    end
end


######################
## Helper functions ##
######################

function load_jld(data_path::AbstractString, data_key::AbstractString, header_key::AbstractString,
     meta_key=nothing, meta_header_key=nothing)
     d = load(data_path)

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


function load_dlm(data_path::AbstractString, meta_path=nothing)
    sep = splitext(data_path)[2] == ".tsv" ? '\t' : ','

    data, header = readdlm(data_path, sep, header=true)
    header = header[:]

    if meta_path != nothing
        meta_data, meta_header = load_dlm(meta_path)
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
        load_biom_hdf5(data_path)
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

end
