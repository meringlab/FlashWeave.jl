module Io

using LightGraphs, SimpleWeightedGraphs
using JLD2, FileIO

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
        ld_results = (nothing, nothing, nothing, nothing)
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
        meta_data, meta_header = readdlm(meta_path, sep, header=true)
        meta_header = meta_header[:]
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
