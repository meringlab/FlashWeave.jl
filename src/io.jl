# note: needed lots of @eval and Base.invokelatest hacks for conditional
# module loading

const valid_net_formats = (".edgelist", ".jld2", ".jld")
const valid_data_formats = (".tsv", ".csv", ".biom", ".jld2", ".jld")

isjld(ext::AbstractString) = ext in (".jld2", ".jld")
isdlm(ext::AbstractString) = ext in (".tsv", ".csv")
isbiom(ext::AbstractString) = ext == ".biom"
isedgelist(ext::AbstractString) = ext == ".edgelist"


"""
    load_data(data_path::AbstractString, meta_path::AbstractString) -> (AbstractMatrix{<:Real}, Vector{String}, AbstractMatrix{<:Real}, Vector{String})

Load tables with OTU count and optionally meta data from disc. Available formats are '.tsv', '.csv', '.biom', '.jld' and '.jld2'.

- `data_path` - path to a file storing an OTU count table

- `meta_data_path` - optional path to a file with meta variable information

- `*_key` - HDF5 keys to access data sets with OTU counts, Meta variables and variable names within a JLD/2 file

- `transposed` - if `true`, rows of `data` are variables and columns are samples
"""
function load_data(data_path::AbstractString, meta_path=nothing; transposed::Bool=false,
     otu_data_key::AbstractString="otu_data", meta_data_key::AbstractString="meta_data",
     otu_header_key::AbstractString="otu_header", meta_header_key::AbstractString="meta_header")
     """Load OTU tables and meta data from various formats.
     -- Set jld keys you don't want to use to 'nothing'
     -- delimited formats must have headers (or row indices if transposed=true)"""
    file_ext = splitext(data_path)[2]
    transposed && file_ext == ".biom" && warn("'transposed' cannot be used with .biom files")

    if isdlm(file_ext)
        ld_results = load_dlm(data_path, meta_path, transposed=transposed)
    elseif isbiom(file_ext)
        ld_results = load_biom(data_path, meta_path)
    elseif isjld(file_ext)
        ld_results = load_jld(data_path, otu_data_key, otu_header_key, meta_data_key, meta_header_key, transposed=transposed)
    else
        error("$(file_ext) not a valid output format. Choose one of $(valid_data_formats)")
    end

    ld_results
end


"""
    save_network(net_path::AbstractString, net_result::FWResult) -> Void

Save network results to disk. Available formats are '.tsv', '.csv', '.jld' and '.jld2'.

- `net_path` - output path for the network

- `net_result` - network results object that should be saved

- `detailed` - output additional information, such as discarding sets, if available
"""
function save_network(net_path::AbstractString, net_result::FWResult; detailed::Bool=false)
    file_ext = splitext(net_path)[2]
    if isedgelist(file_ext)
        write_edgelist(net_path, graph(net_result))
    elseif isjld(file_ext)
        # isdefined(:FileIO) || @eval using FileIO: save, load
        # Base.invokelatest(save, net_path, "results", net_result)
        save(net_path, "results", net_result)
    else
        error("$(file_ext) not a valid output format. Choose one of $(valid_net_formats)")
    end

    if detailed
        out_trunk = splitext(net_path)[1]
        save_rejections(out_trunk * "_rejections.tsv", net_result)
        save_unfinished_variable_info(out_trunk * "_unchecked.tsv", net_result)
    end
end


"""
    load_network(net_path::AbstractString) -> FWResult{Int}

Load network results from disk. Available formats are '.tsv', '.csv', '.jld' and '.jld2'.

- `net_path` - path from which to load the network results
"""
function load_network(net_path::AbstractString)
    file_ext = splitext(net_path)[2]
    if isedgelist(file_ext)
        G = read_edgelist(net_path)
        net_result = FWResult(G)

    elseif isjld(file_ext)
        # isdefined(:FileIO) || @eval using FileIO: save, load
        # d = Base.invokelatest(load, net_path)
        # net_result = d["results"]
        net_result = load(net_path, "results")

    else
        error("$(file_ext) not a valid network format. Valid formats are $(valid_net_formats)")
    end
end


######################
## Helper functions ##
######################

function load_jld(data_path::AbstractString, otu_data_key::AbstractString, otu_header_key::AbstractString,
     meta_data_key=nothing, meta_header_key=nothing; transposed::Bool=false)
     # isdefined(:FileIO) || @eval using FileIO: save, load
     # d = Base.invokelatest(load, data_path)
     d = load(data_path)

     data = d[otu_data_key]
     header = d[otu_header_key]

     if meta_data_key != nothing
         meta_data = d[meta_data_key]
         meta_header = d[meta_header_key]
     else
         meta_data = meta_header = nothing
     end

     if transposed
         data = data'
         meta_data = meta_data'
     end

     data, header, meta_data, meta_header
 end


hasrowids(data::AbstractMatrix) = data[1, 1] == "" || isa(data[2, 1], AbstractString)

# this could eventually replaced with FileIO
function load_dlm(data_path::AbstractString, meta_path=nothing; transposed::Bool=false)
    sep = splitext(data_path)[2] == ".tsv" ? '\t' : ','
    data = readdlm(data_path, sep)

    if transposed
        # hacky transpose for string data
        data = permutedims(data, (2,1))
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
