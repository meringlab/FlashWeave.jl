function factors_to_ints(x::AbstractVector)
    # need more complicated check because x can be AbstractVector{Any}
    if isa(x[1], AbstractString)
        factor_map = Dict([(xi, fi) for (fi, xi) in enumerate(sort(unique(x)))])
        Int[factor_map[xi] for xi in x]
    else
        x
    end
end


function factors_to_ints(X::AbstractMatrix)
    M = issparse(X) ? SparseMatrixCSC : Matrix
    M{Float64}(mapslices(factors_to_ints, X; dims=1))
end


function check_onehot(x::AbstractVector)
    # check if proper floating point vector
    if isa(x[1], AbstractFloat) && any(!iszero(xi % 1) for xi in x)
        return false, []
    # otherwise check number of categories
    else
        categories = sort(unique(x))
        length(categories) > 2, categories
    end
end


function onehot(x::AbstractVector, var_name="", check::Bool=true)
    needs_onehot, categories = check_onehot(x)
    if !check || needs_onehot
        cols_enc = Vector{Int}[]
        vnames_enc = String[]

        for category in categories
            push!(cols_enc, Vector{Int}(x .== category))
            if !isempty(var_name)
                push!(vnames_enc, var_name * "_" * string(category))
            end
        end
    else
        cols_enc, vnames_enc = [factors_to_ints(x)], [var_name]
    end

    hcat(cols_enc...), vnames_enc
end


function onehot(X::AbstractMatrix, vnames::AbstractVector{<:AbstractString}=String[];
    check::Bool=true, verbose::Bool=true)

    enc_results = [onehot(X[:, i], isempty(vnames) ? "" : vnames[i], check) for i in 1:size(X, 2)]

    if verbose
        enc_mask = [size(X, 2) > 1 for (X, vn) in enc_results]
        num_enc = sum(enc_mask)
        enc_vnames = isempty(vnames) ? [] : vnames[enc_mask]
        enc_vname_str = isempty(enc_vnames) ? "" : " (" * join(enc_vnames, ", ") * ")"

        pl_str1 = num_enc == 1 ? "" : "s"
        pl_str2 = num_enc == 1 ? "it" : "them"
        num_enc > 0 && @warn "$num_enc factor variable$(pl_str1) with more than two categories were detected$(enc_vname_str), splitting $(pl_str2) into separate dummy variables (One Hot)"
    end

    M = issparse(X) ? SparseMatrixCSC : Matrix
    X_enc = hcat(map(first, enc_results)...) |> M{Float64}

    if !isempty(vnames)
        vnames_enc = vcat(map(x->x[2], enc_results)...) |> Vector{String}
    else
        vnames_enc = vnames
    end

    X_enc, vnames_enc
end


function mapslices_sparse_nz(f, A::SparseMatrixCSC, dim::Integer=1)
    if dim == 1
        A = permutedims(A)
    end
    result_vec = zeros(eltype(A), size(A, 2))
    for j in 1:size(A, 2)
        col_vec = A[:, j]
        result_vec[j] = f(col_vec.nzval)
    end
    result_vec
end


function pseudocount_vars_from_sample(s::AbstractVector{ElType}) where ElType <: AbstractFloat
    z_mask = s .== 0
    k = sum(z_mask)
    Nprod = sum(log.(s[.!z_mask]))
    p = length(s)
    return k, Nprod, p
end


function adaptive_pseudocount(x1::ElType, s1::AbstractVector{ElType}, s2::AbstractVector{ElType}) where ElType <: AbstractFloat
    k, Nprod1_log, p = pseudocount_vars_from_sample(s1)
    adaptive_pseudocount(x1, k, Nprod1_log, p, s2)
end


function adaptive_pseudocount(x1::ElType, k::Integer, Nprod1_log::AbstractFloat, p::Integer,
    s2::AbstractVector{ElType}) where ElType <: AbstractFloat
    n, Nprod2_log, _ = pseudocount_vars_from_sample(s2)
    @assert n < p && k < p "samples with all zero abundances are not allowed"
    x2_log = (1 / (n-p)) * ((k-p)*log(x1) + Nprod1_log - Nprod2_log)
    return exp(x2_log)
end


function adaptive_pseudocount!(X::Matrix{ElType}) where ElType <: AbstractFloat
    max_depth_index = findmax(sum(X, dims=2))[2][1]
    max_depth_sample = view(X, max_depth_index, :)
    min_abund = minimum(X[X .!= 0])
    base_pcount = min_abund >= 1 ? 1.0 : min_abund / 10
    max_depth_pvars = pseudocount_vars_from_sample(max_depth_sample)
    pseudo_counts = [adaptive_pseudocount(base_pcount, max_depth_pvars..., view(X, x, :)) for x in 1:size(X, 1)]
    pcount_z_mask = pseudo_counts .== 0
    if any(pcount_z_mask)
        @warn "adaptive pseudo-counts for $(sum(pcount_z_mask)) samples were lower than machine precision due to insufficient counts, removing them"
        X = X[.!pcount_z_mask, :]
        pseudo_counts = pseudo_counts[.!pcount_z_mask]
    end

    for i in 1:size(X, 1)
        s_vec = @view X[i, :]
        s_vec[s_vec .== 0] .= pseudo_counts[i]
    end
    X
end

function clr!(X::SparseMatrixCSC{ElType}) where ElType <: AbstractFloat
    """Specialized in-place version for sparse matrices that always excludes zero entries (thereby no need for pseudo counts)"""
    gmeans_vec = mapslices_sparse_nz(geomean, X, 1)
    rows = rowvals(X)

    for j in 1:size(X, 2)
        for i in nzrange(X, j)
            row_gmean = gmeans_vec[rows[i]]
            X.nzval[i] = log(X.nzval[i] / row_gmean)
        end
    end
end


function clr!(X::Matrix{ElType}; pseudo_count::ElType=1e-5, ignore_zeros::Bool=false) where ElType <: AbstractFloat

    if !ignore_zeros
        X .+= pseudo_count
        center_fun = geomean
    else
        center_fun = x -> geomean(x[x .!= 0.0])
    end

    X .= log.(X ./ mapslices(center_fun, X, dims=2))

    if ignore_zeros
        X[isinf.(X)] .= 0.0
    end
    nothing
end


function adaptive_clr!(X::Matrix{ElType}) where ElType <: AbstractFloat
    X = adaptive_pseudocount!(X)
    clr!(X, pseudo_count=0.0, ignore_zeros=false)
    X
end


function discretize(X::AbstractMatrix{ElType}; n_bins::Integer=3, nz::Bool=true,
        rank_method::String="tied", disc_method::String="median", nz_mask::BitMatrix=BitMatrix(undef, (0,0))) where ElType <: AbstractFloat
    if nz
        if issparse(X)
            disc_vecs = SparseVector{Int}[]
            for j in 1:size(X, 2)
                push!(disc_vecs, discretize_nz(X[:, j], n_bins, rank_method=rank_method, disc_method=disc_method))
            end
            return hcat(disc_vecs...)
        else
            disc_vecs = [discretize_nz(view(X, :, j), view(nz_mask, :, j), n_bins, rank_method=rank_method, disc_method=disc_method) for j in 1:size(X, 2)]
            return hcat(disc_vecs...)
        end
    else
        return mapslices(x -> discretize(x, n_bins, rank_method=rank_method, disc_method=disc_method), X, 1)
    end
end

function discretize(x_vec::Vector{ElType}, n_bins::Integer=3; rank_method::String="tied", disc_method::String="median") where ElType <: AbstractFloat
    if disc_method == "median"
        if isempty(x_vec)
            disc_vec = x_vec
        else
            if rank_method == "dense"
                x_vec = denserank(x_vec)
            elseif rank_method == "tied"
                x_vec = tiedrank(x_vec)
            else
                error("$rank_method not a valid ranking method")
            end

            x_vec /= maximum(x_vec)

            # compute step, add small number to avoid a separate bin for rank 1.0
            step = (1.0 / n_bins) + 1e-5
            disc_vec = map(x -> Int(floor((x) / step)), x_vec)
        end

    elseif disc_method == "mean"
        if n_bins > 2
            error("disc_method $disc_method only works with 2 bins.")
        end

        bin_thresh = mean(x_vec)
        disc_vec = map(x -> x <= bin_thresh ? 0 : 1, x_vec)
    else
        error("$disc_method is not a valid discretization method.")
    end

    disc_vec
end


function discretize_nz(x_vec::SparseVector{ElType}, n_bins::Integer=3;
        rank_method::String="tied", disc_method::String="median") where ElType <: AbstractFloat
    disc_nz_vec = discretize(x_vec.nzval, n_bins-1, rank_method=rank_method, disc_method=disc_method) .+ 1
    SparseVector(x_vec.n, x_vec.nzind, disc_nz_vec)
end

function discretize_nz(x_vec::AbstractVector{ElType}, nz_vec::AbstractVector{Bool}, n_bins::Integer=3; rank_method::String="tied", disc_method::String="median") where ElType <: AbstractFloat
    if any(nz_vec)
        x_vec_nz = x_vec[nz_vec]
        disc_nz_vec = discretize(x_vec_nz, n_bins-1, rank_method=rank_method, disc_method=disc_method) .+ 1
        disc_vec = zeros(Int, size(x_vec))
        disc_vec[nz_vec] = disc_nz_vec
    else
        disc_vec = zeros(ElType, size(x_vec))
    end

    disc_vec
end


iscontinuousnorm(norm::String) = norm == "rows" || startswith(norm, "clr")

function discretize_meta!(meta_data::Matrix{ElType}, norm, n_bins) where ElType <: Real
    for i in 1:size(meta_data, 2)
        meta_vec = meta_data[:, i]
        try
            disc_meta_vec = convert(Vector{Int}, meta_vec)
            if norm == "clr_nz"
                disc_meta_vec += 1
            end
            meta_vec = convert(Vector{ElType}, disc_meta_vec)
        catch InexactError
            if !iscontinuousnorm(norm)
                disc_meta_vec = discretize(meta_vec, n_bins)
                meta_vec = convert(Vector{ElType}, disc_meta_vec)
            end
        end
        meta_data[:, i] .= meta_vec
    end
end

function discretize_meta(meta_data::SparseMatrixCSC{ElType}, norm, n_bins) where ElType <: Real
    meta_data_dense = Matrix(meta_data)
    discretize_meta!(meta_data_dense, norm, n_bins)
    sparse(meta_data_dense)
end


function clrnorm(data::AbstractMatrix, norm::String, clr_pseudo_count::AbstractFloat)
    """Covers all flavors of clr transform, makes sparse matrices dense if pseudo-counts
    are used to make computations more efficient"""

    if norm == "clr"
        data = convert(Matrix{Float64}, data)
        clr!(data, pseudo_count=clr_pseudo_count)
    elseif norm == "clr_adapt"
        data = convert(Matrix{Float64}, data)
        data = adaptive_clr!(data)
    elseif norm == "clr_nz"
        if issparse(data)
            data = convert(SparseMatrixCSC{Float64}, data)
            clr!(data)
        else
            data = convert(Matrix{Float64}, data)
            clr!(data, pseudo_count=0.0, ignore_zeros=true)
        end
    end

    data
end

rownorm!(X::Matrix{ElType}) where ElType <: AbstractFloat = X ./= sum(X, dims=2)

function rownorm!(X::SparseMatrixCSC{ElType}) where ElType <: AbstractFloat
    """Specialized in-place version for sparse matrices"""
    sum_vec = sum(X, dims=2)
    rows = rowvals(X)

    for i in 1:size(X, 2)
        for j in nzrange(X, i)
            row_sum = sum_vec[rows[j]]
            X.nzval[j] = X.nzval[j] / row_sum
        end
    end
end


presabs_norm!(X::SparseMatrixCSC{ElType}) where ElType <: Real = map!(sign, X.nzval, X.nzval)
presabs_norm!(X::Matrix{ElType}) where ElType <: Real = map!(sign, X, X)


function preprocess_data(data::AbstractMatrix, norm::String; clr_pseudo_count::AbstractFloat=1e-5,
    n_bins::Integer=3, rank_method::String="tied",
    disc_method::String="median", verbose::Bool=true, meta_mask::BitVector=falses(size(data, 2)),
    make_sparse::Bool=issparse(data),
    prec::Integer=32, filter_data=true, header::Vector{String}=String[], make_onehot::Bool=true)

    verbose && println("Removing variables with 0 variance (or equivalently 1 level) and samples with 0 reads")
    has_meta = any(meta_mask)

    if has_meta
        meta_data = data[:, meta_mask]
        nometa_mask = .!meta_mask
        data = data[:, nometa_mask]

        if !isempty(header)
            meta_header = header[meta_mask]
            header = header[nometa_mask]
        else
            meta_header = nothing
        end

        # one-hot encode meta data and convert string factors to ints
        if make_onehot
            meta_data, meta_header = onehot(meta_data, meta_header; verbose=verbose)
        else
            @warn "Skipping one-hot encoding, only experts should choose this option."
            meta_data = factors_to_ints(meta_data)
        end
    end

    data, make_sparse = check_convert_sparse(data, make_sparse, norm, prec)
    M = make_sparse ? SparseMatrixCSC : Matrix

    if filter_data
        unfilt_dims = size(data)
        col_mask = (var(data, dims=1)[:] .> 0.0)[:]
        data = data[:, col_mask]
        row_mask = (sum(data, dims=2)[:] .> 0)[:]
        data = data[row_mask, :]
        if has_meta
            meta_data = meta_data[row_mask, :]
        end

        if !isempty(header)
            header = header[col_mask]
        end
    else
        row_mask = nothing
    end

    if verbose
        println("Discarded ", unfilt_dims[1] - size(data, 1), " samples and ", unfilt_dims[2] - size(data, 2), " variables.")
        println("\nNormalization")
    end

    if norm == "rows"
        rownorm!(data)

    elseif startswith(norm, "clr")
        data = clrnorm(data, norm, clr_pseudo_count)

    elseif norm == "binary"
        presabs_norm!(data)
        data = M{Int}(data)

        unreduced_vars = size(data, 2)
        bin_mask =  (get_levels(data) .== 2)[:]
        data = data[:, bin_mask]

        if !isempty(header)
            header = header[bin_mask]
        end

        verbose && println("\t-> removed $(unreduced_vars - size(data, 2)) variables with not exactly 2 levels")

    elseif startswith(norm, "binned")
        if startswith(norm, "binned_nz")
            # to make sure, non-absences that become zero after
            # pre-normalization are not considered as absences
            nz_mask = issparse(data) ? BitMatrix(undef, (0, 0)) : data .!= 0

            if endswith(norm, "rows")
                rownorm!(data)
            elseif endswith(norm, "clr")
                data = clrnorm(data, "clr_nz", 0.0)
            end
            data = discretize(data, n_bins=n_bins, nz=true, rank_method=rank_method, disc_method=disc_method,
            nz_mask=nz_mask)
        else
            data = discretize(data, n_bins=n_bins, nz=false, rank_method=rank_method, disc_method=disc_method)
        end

        unreduced_vars = size(data, 2)
        bin_mask =  (get_levels(data) .== n_bins)[:]
        data = data[:, bin_mask]

        if !isempty(header)
            header = header[bin_mask]
        end

        verbose && println("\t-> removed $(unreduced_vars - size(data, 2)) variables with less than $n_bins levels")

    else
        error("$norm not a valid normalization method.")
    end

    if has_meta
        if issparse(meta_data)
            meta_data = discretize_meta(meta_data, norm, 2)
        else
            discretize_meta!(meta_data, norm, 2)
        end
        meta_mask = vcat(falses(size(data, 2)), trues(size(meta_data, 2)))
        data = hcat(data, convert(typeof(data), meta_data))

        if !isempty(header)
            append!(header, meta_header)
        end
    else
        meta_mask = falses(size(data, 2))
    end

    target_base_type_str = iscontinuousnorm(norm) ? "Float" : "Int"
    T = eval(Symbol("$target_base_type_str$prec"))

    # need two-step conversion because direct 64+Dense -> 32+sparse currently
    # does not work
    M_out = make_sparse ? SparseMatrixCSC : Matrix
    data = data |> M{T} |> M_out{T}

    (data=data, header=header, meta_mask=meta_mask, obs_filter_mask=row_mask)
end


function preprocess_data_default(data::AbstractMatrix, test_name::AbstractString; verbose::Bool=true,
     make_sparse::Bool=issparse(data), make_onehot::Bool=true, meta_mask::BitVector=falses(size(data, 2)),
     prec::Integer=32, header::Vector{String}=String[], preprocess_kwargs...)
    default_norm_dict = Dict("mi" => "binary",
                             "mi_nz" => "binned_nz_clr",
                             "fz" => "clr_adapt",
                             "fz_nz" => "clr_nz",
                             "mi_expdz" => "binned_nz_clr")
    preprocess_data(data, default_norm_dict[test_name]; verbose=verbose, make_sparse=make_sparse,
    make_onehot=make_onehot, meta_mask=meta_mask, prec=prec, header=header, preprocess_kwargs...)
end


function check_convert_sparse(data, make_sparse, norm_str, prec)
    T = try
            eval(Symbol("Float$prec"))
        catch UndefVarError
            error("'$prec' not a valid precision")
    end

    if make_sparse && (norm_str == "clr_adapt"  || norm_str == "fz")
        @warn "Adaptive CLR is inefficient with sparse data, using dense format"
        make_sparse = false
    end

    MatType = make_sparse ? SparseMatrixCSC{T, Int64} : Matrix{T}
    data = convert(MatType, data)
    data, make_sparse
end

"""
    normalize_data(data::AbstractMatrix{<:Real}) -> AbstractMatrix OR (AbstractMatrix{<:Real}, Vector{String})

Normalize data using various forms of clr transform and discretization. This should only be used manually when experimenting with different normalization techniques.

- `data` - data table with information on OTU counts and (optionally) meta variables

- `header` - names of variable-column s in `data`

- `meta_mask` - true/false mask indicating which variables are meta variables

- `test_name` - name of a FlashWeave-specific statistical test mode, the appropriate normalization method will be chosen automatically

- `norm_mode` - identifier of a valid normalization mode ('clr-adapt', 'clr-nonzero', 'clr-nonzero-binned', 'pres-abs', 'tss', 'tss-nonzero-binned')

- `filter_data` - whether to remove samples and variables without information from `data`

- `verbose` - print progress information

- `prec` - precision in bits to use for calculations (16, 32, 64 or 128)

- `make_sparse`, `make_onehot` - see docstring for "learn_network()"

"""
function normalize_data(data::AbstractMatrix; test_name::AbstractString="", norm_mode::AbstractString="",
    header::Vector{String}=String[], meta_mask::BitVector=falses(size(data, 2)),
    verbose::Bool=true, prec::Integer=32, filter_data::Bool=true, make_sparse::Bool=true, make_onehot::Bool=true)
    @assert xor(isempty(test_name), isempty(norm_mode)) "provide either test_name and norm_mode (but not both)"

    mode_map = Dict("clr-adapt"=>"clr_adapt", "clr-nonzero"=>"clr_nz",
                    "clr-nonzero-binned"=>"binned_nz_clr", "pres-abs"=>"binary",
                    "tss"=>"rows", "tss-nonzero-binned"=>"binned_nz_rows")

    @assert isempty(norm_mode) || haskey(mode_map, norm_mode) "$norm_mode not a valid normalization mode"
    @assert xor(test_name == "", norm_mode == "") "provide exactly one out of 'test_name' and 'norm_mode'"

    if !isempty(test_name)
        preproc_fun = preprocess_data_default
        norm_str = test_name
    else
        preproc_fun = preprocess_data
        norm_str = mode_map[norm_mode]
    end

    norm_results = preproc_fun(data, norm_str; meta_mask=meta_mask, header=header,
                               verbose=verbose, filter_data=filter_data, prec=prec, make_sparse=make_sparse,
                               make_onehot=make_onehot)
end
