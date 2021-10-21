# note: needed lots of @eval and Base.invokelatest hacks for conditional
# module loading

const valid_net_formats = (".edgelist", ".gml", ".jld2")
const valid_data_formats = (".tsv", ".csv", ".biom", ".jld2")
const valid_dlm_formats = (".tsv", ".csv")

isjld(ext::AbstractString) = ext == ".jld2"
isdlm(ext::AbstractString) = ext in valid_dlm_formats
isbiom(ext::AbstractString) = ext == ".biom"
isedgelist(ext::AbstractString) = ext == ".edgelist"
isgml(ext::AbstractString) = ext == ".gml"
isdefaultkey(key::AbstractString) = key in ("otu_data", "otu_header", "meta_data", "meta_header")


"""
    load_data(data_path::AbstractString, meta_path::AbstractString) -> (AbstractArray{<:Real, 2}, Vector{String}, AbstractArray{<:Real, 2}, Vector{String})

Load matrices with OTU count and optionally meta data from disc. Available formats are '.tsv', '.csv', '.biom' and '.jld2'.

- `data_path` - path to a file storing an OTU count matrix

- `meta_data_path` - optional path to a file with meta variable information

- `*_key` - HDF5 keys to access data sets with OTU counts, Meta variables and variable names in a JLD2 file. If a data item is absent the corresponding key should be 'nothing'. See '?load_data' for additional information.

- `transposed` - if `true`, rows of `data` are variables and columns are samples
"""
function load_data(data_path::AbstractString, meta_path::StrOrNoth=nothing; transposed::Bool=false,
     otu_data_key::StrOrNoth="otu_data", meta_data_key::StrOrNoth="meta_data",
     otu_header_key::StrOrNoth="otu_header", meta_header_key::StrOrNoth="meta_header")
     """Load OTU tables and meta data from various formats.
     -- Set jld2 keys you don't want to use to 'nothing'
     -- delimited formats must have headers (or row indices if transposed=true)"""
    file_ext = splitext(data_path)[2]
    transposed && file_ext == ".biom" && @warn("'transposed' cannot be used with .biom files")

    if !isnothing(meta_path)
        meta_file_ext = splitext(meta_path)[2]
        !isdlm(meta_file_ext) && @error "$(meta_file_ext) is an invalid meta data format, please provide one of $(valid_dlm_formats)"
    end

    if isdlm(file_ext)
        ld_results = load_dlm(data_path, meta_path, transposed=transposed)
    elseif isbiom(file_ext)
        ld_results = load_biom(data_path, meta_path)
    elseif isjld(file_ext)
        @warn "jld2 support is deprecated and will be removed in future versions of FlashWeave"
        !isnothing(meta_path) && @error "jld2 format not compatible with external meta data files, please add meta data directly to the jld2 file (default group key: 'meta_data')"
        for (key, desc) in [(otu_data_key, "otu data"), (otu_header_key, "otu ids")]
            @assert !isnothing(key) "must provide a key for $(desc)"
        end
        ld_results = load_jld(data_path, otu_data_key, otu_header_key, meta_data_key, meta_header_key, transposed=transposed)
    else
        error("$(file_ext) not a valid output format. Choose one of $(valid_data_formats)")
    end

    ld_results
end


"""
    save_network(net_path::AbstractString, net_result::FWResult) -> Void

Save network results to disk. Available formats are '.edgelist', '.gml' and '.jld2'.

- `net_path` - output path for the network

- `net_result` - network results object that should be saved

- `detailed` - save additional information, such as discarding sets, if available (output file suffixes: '_rejections', '_unchecked')
"""
function save_network(net_path::AbstractString, net_result::FWResult; detailed::Bool=false)
    file_ext = splitext(net_path)[2]
    if isedgelist(file_ext)
        write_edgelist(net_path, net_result)
    elseif isgml(file_ext)
        write_gml(net_path, net_result)
    elseif isjld(file_ext)
        @warn "jld2 support is deprecated and will be removed in future versions of FlashWeave"
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

Load network results from disk. Available formats are '.edgelist', '.gml' and '.jld2'. For GML, only files with structure identical to save_network('network.gml') output can currently be loaded. FlashWeave parameters that were used for network inference are only available when loading from JLD2.

- `net_path` - path from which to load the network results
"""
function load_network(net_path::AbstractString)
    file_ext = splitext(net_path)[2]
    if isedgelist(file_ext)
        net_result = read_edgelist(net_path)
    elseif isgml(file_ext)
        net_result = read_gml(net_path)
    elseif isjld(file_ext)
        net_result = load(net_path, "results")
    else
        error("$(file_ext) not a valid network format. Valid formats are $(valid_net_formats)")
    end
end


######################
## Helper functions ##
######################

function load_jld(data_path::AbstractString, otu_data_key::AbstractString, otu_header_key::AbstractString,
     meta_data_key::StrOrNoth=nothing, meta_header_key::StrOrNoth=nothing; transposed::Bool=false)
     d = load(data_path)

     for (key, key_desc) in [(otu_data_key, "otu_data_key"),
                             (otu_header_key, "otu_header_key"),
                             (meta_data_key, "meta_data_key"),
                             (meta_header_key, "meta_header_key")]
         !isnothing(key) && !haskey(d, key) && !(key in ["meta_data_key", "meta_header_key"]) && error("key '$key' not found in input file. Please make sure to provide the appropriate $key_desc when using non-standard input files for FlashWeave or set $key_desc to 'nothing'. Keys present in input file: $(join(keys(d), ", "))")
     end

     data = d[otu_data_key]
     header = d[otu_header_key]

     if !isnothing(meta_data_key) && haskey(d, meta_data_key)
         meta_data = d[meta_data_key]
         meta_header = d[meta_header_key]
     else
         meta_data = meta_header = nothing
     end

     if transposed
         data = data'
         if !isnothing(meta_data)
             meta_data = meta_data'
         end
     end

     data, header, meta_data, meta_header
 end


hasrowids(data::AbstractMatrix, header::AbstractVector) =
    header[1] == "" || (length(unique(data[:, 1])) == size(data, 1) && (isa(data[1, 1], String) || isa(data[1, 1], SubString)))

# this could eventually be replaced with FileIO
function load_dlm(data_path::AbstractString, meta_path=nothing; transposed::Bool=false, type_data::Bool=true)
    sep = splitext(data_path)[2] == ".tsv" ? '\t' : ','

    # to support 'transposed', we separate the header later instead of using
    # the flag here
    data_raw = readdlm(data_path, sep)

    if transposed
        # hacky transpose for string data
        data_raw = permutedims(data_raw, (2,1))
    end

    header_raw = data_raw[1, :]
    data_raw = data_raw[2:end, :]

    if hasrowids(data_raw, header_raw)
        data_raw = data_raw[:, 2:end]
        header_raw = header_raw[2:end]
    end

    header = string.(header_raw)[:]

    # special case encountered in the wild (see test "numeric IDs")
    if all(endswith.(header, ".0"))
        header = map(x -> x[1:end-2], header)
    end

    data = type_data ? Matrix{Float64}(data_raw) : data_raw

    if !isnothing(meta_path)
        meta_data, meta_header = load_dlm(meta_path, transposed=transposed, type_data=false)
    else
        meta_data = meta_header = nothing
    end

    data, header, meta_data, meta_header
end


function load_biom_json(data_path)
    json_struc = JSON.parsefile(data_path)
    otu_table = Matrix{Int}(hcat(json_struc["data"]...))

    if json_struc["matrix_type"] == "sparse"
        otu_table = permutedims(otu_table)
        otu_table = permutedims(sparse(otu_table[:, 1] .+ 1, otu_table[:, 2] .+ 1, otu_table[:, 3]))
    end

    header = [x["id"] for x in json_struc["rows"]]
    otu_table, header
end


function load_biom_hdf5(data_path)
    f = h5open(data_path, "r")
    # use either 'attrs' or 'attributes' (depends on HDF5 version)
    attr_fun = isdefined(HDF5, :attrs) ? attrs : attributes
    m, n = read(attr_fun(f)["shape"])
    colptr, rowval, nzval = [read(f, "sample/matrix/$key") for key in ["indptr", "indices", "data"]]
    otu_table = permutedims(SparseMatrixCSC(m, n, colptr .+ 1, rowval .+ 1, Vector{Int}(nzval)))
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
            error("Error in 'load_biom'. File $data_path seems not to be valid .biom")
        end
    end

    if !isnothing(meta_path)
        meta_data, meta_header = load_dlm(meta_path, type_data=false)
    else
        meta_data = meta_header = nothing
    end

    data, header, meta_data, meta_header
end


function load_rejections(rej_path)
    rej_dict = Dict{Int, RejDict{Int}}()

    open(rej_path, "r") do f
        # skip header
        readline(f)

        for line in eachline(f)
            line_items = split(line, "\t")
            var_A, var_B = parse.(Int, split(line_items[1], " <-> "))
            Zs = parse.(Int, Tuple(split(line_items[2], ",")))
            stat, pval = parse.(Float64, line_items[3:4])
            n_tests = parse(Int, line_items[5])
            frac_tested = parse(Float64, line_items[6])
            df = parse(Int, line_items[7])
            suff_power = parse(Bool, line_items[8])
            rej_tuple = (Zs, TestResult(stat, pval, df, suff_power), (n_tests, frac_tested))
            if haskey(rej_dict, var_A)
                rej_dict[var_A][var_B] = rej_tuple
            else
                rej_dict[var_A] = Dict(var_B=>rej_tuple)
            end
        end
    end
    rej_dict
end


function load_unfinished_variable_info(unf_path)
    unf_dict = Dict()

    open(unf_path, "r") do f
        # skip header
        readline(f)

        for line in eachline(f)
            line_items = split(line, "\t")
            var_A = parse(Int, line_items[1])
            phase = line_items[2][1]
            unf_vars = parse.(Int, split(line_items[3], ","))
            unf_dict[var_A] = (phase=phase, unchecked_vars=unf_vars)
        end
    end
    unf_dict
end


function save_rejections(rej_path, net_result; digits=5)
    rej_dict = rejections(net_result)

    open(rej_path, "w") do f
        if isempty(rej_dict)
            write(f, "# No rejections found, you may have forgotten to specify 'track_rejections' when running FlashWeave")
        else
            write(f, join(["Edge", "Rejecting_set", "Stat", "P_value", "Num_tests", "Perc_tested", "Df", "SuffPower"], "\t"), "\n")
            for (var_A, rej_nbr_dict) in rej_dict
                for (var_B, rej_info) in rej_nbr_dict
                    test_res = rej_info[2]
                    stat, pval, df, suff_power = [getproperty(test_res, p) for p in [:stat, :pval, :df, :suff_power]]
                    rej_set = rej_info[1]
                    num_tests, frac_tested = rej_info[3]

                    line_items = [string(var_A) * " <-> " * string(var_B), join(rej_set, ","), round(stat, digits=digits),
                                  round(pval, digits=digits), num_tests, round(frac_tested, digits=digits), df, suff_power]
                    write(f, join(line_items, "\t"), "\n")
                end
            end
        end
    end
end


function save_unfinished_variable_info(unf_path, net_result)
    unf_states_dict = unfinished_states(net_result)
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


function write_edgelist(out_path::AbstractString, net_result::FWResult)
    G = graph(net_result)
    meta_mask = meta_variable_mask(net_result)
    header = names(net_result)

    open(out_path, "w") do out_f
        write(out_f, "# header\t", join(header, ","), "\n")
        write(out_f, "# meta mask\t", join(meta_mask, ","), "\n")

        for e in edges(G)
            if isnothing(header)
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


function read_edgelist(in_path::AbstractString)
    srcs = Int[]
    dsts = Int[]
    ws = Float64[]

    header, meta_mask = open(in_path, "r") do in_f
        header_items = split(readline(in_f), "\t")[end]
        header = Vector{String}(split(header_items, ","))
        inv_header = Dict{eltype(header), Int}(zip(header, 1:length(header)))

        meta_items = split(readline(in_f), "\t")[end]
        meta_mask = BitVector(map(x->parse(Bool, x), split(meta_items, ",")))

        for line in eachline(in_f)
            line_items = split(chomp(line), '\t')

            src = inv_header[line_items[1]]
            dst = inv_header[line_items[2]]

            push!(srcs, src)
            push!(dsts, dst)
            push!(ws, parse(Float64, line_items[end]))
        end

        header, meta_mask
    end
    G = SimpleWeightedGraph_nodemax(srcs, dsts, ws; m=length(header))
    FWResult(G; variable_ids=header, meta_variable_mask=meta_mask)
end


function write_gml(out_path::AbstractString, net_result::FWResult)
    G = graph(net_result)
    header = names(net_result)
    meta_mask = net_result.meta_variable_mask

    open(out_path, "w") do out_f
        write(out_f, "graph [", "\n")
        write(out_f, "\tdirected 0", "\n")

        for node in vertices(G)
            write(out_f, "\tnode [", "\n")
            write(out_f, "\t\tid " * string(node), "\n")
            write(out_f, "\t\tlabel \"" * header[node] * "\"", "\n")
            write(out_f, "\t\tmv " * string(Int(meta_mask[node])), "\n")
            write(out_f, "\t]", "\n")
        end

        for e in edges(G)
            e1, e2, weight = e.src, e.dst, e.weight
            write(out_f, "\tedge [", "\n")
            write(out_f, "\t\tsource " * string(e1), "\n")
            write(out_f, "\t\ttarget " * string(e2), "\n")
            write(out_f, "\t\tweight " * string(weight), "\n")
            write(out_f, "\t]", "\n")
        end

        write(out_f, "]", "\n")
    end
    nothing
end


function parse_gml_field(in_f::IO)
    line = strip(readline(in_f))
    info_pairs = Tuple[]

    if !(startswith(line, "node") || startswith(line, "edge"))
        return info_pairs
    end

    if startswith(line, "node") || startswith(line, "edge")
        while !startswith(line, "]")
            push!(info_pairs, Tuple(split(line)))
            line = strip(readline(in_f))
        end
    end

    info_pairs
end


function read_gml(in_path::AbstractString)
    node_dict = Dict{Int,Vector{Tuple}}()

    srcs = Int[]
    dsts = Int[]
    ws = Float64[]

    header, meta_mask = open(in_path, "r") do in_f
        line = readline(in_f)
        line = readline(in_f)

        node_info = parse_gml_field(in_f)
        while node_info[1][1] == "node"
            node_id = parse(Int, node_info[2][2])
            node_dict[node_id] = node_info
            node_info = parse_gml_field(in_f)
        end

        n_nodes = maximum(keys(node_dict))
        header = fill("", n_nodes)
        meta_mask = falses(n_nodes)
        for (node_id, n_inf) in node_dict
            header[node_id] = n_inf[3][2][2:end-1]
            meta_mask[node_id] = Bool(parse(Int, n_inf[4][2]))
        end

        edge_info = node_info
        while !isempty(edge_info) && edge_info[1][1] == "edge"
            push!(srcs, parse(Int, edge_info[2][2]))
            push!(dsts, parse(Int, edge_info[3][2]))
            push!(ws, parse(Float64, edge_info[4][2]))
            edge_info = parse_gml_field(in_f)
        end

        header, meta_mask
    end

    G = SimpleWeightedGraph(srcs, dsts, ws)
    net_result = FWResult(G; variable_ids=header, meta_variable_mask=meta_mask)
end
