function pw_unistat_matrix{T,ElType<:Real}(data::AbstractMatrix{ElType}, test_name::String; parallel::String="single",
        pw_stat_dict::Dict{Int,OrderedDict{Int,Tuple{Float64,Float64}}}=Dict{Int,OrderedDict{Int,Tuple{Float64,Float64}}}(),
    pw_uni_args::Dict{Symbol,T}=Dict{Symbol,Any}(), unsig_is_na::Bool=true)

    if isempty(pw_stat_dict)
        pw_stat_dict = pw_univar_neighbors(data; test_name=test_name, parallel=parallel, pw_uni_args...)
    end

    if unsig_is_na
        stat_mat = repmat([NaN64], size(data, 2), size(data, 2))
    else
        stat_mat = zeros(Float64, size(data, 2), size(data, 2))
    end

    for var_A in keys(pw_stat_dict)
        for var_B in keys(pw_stat_dict[var_A])
            (stat, pval) = pw_stat_dict[var_A][var_B]
            stat_mat[var_A, var_B] = stat
        end
    end
    stat_mat
end


function cluster_hierarchical(dist_mat, data, cluster_sim_threshold)
    clust_dict = Dict{Int,Set{Int}}()
    clust_obj = hclust(dist_mat, :average)
    clusts = cutree(clust_obj, h=1.0 - cluster_sim_threshold)

    var_sizes = sum(data, 1)

    for clust_id in unique(clusts)
        clust_members = findn(clusts .== clust_id)
        if length(clust_members) > 1
            clust_sizes = @view var_sizes[clust_members]
            clust_repres = clust_members[findmax(clust_sizes)[2]]
        else
            clust_repres = clust_members[1]
        end
        clust_dict[clust_repres] = Set(clust_members)
    end
    clust_dict
end


function cluster_data{T,ElType<:Real}(data::AbstractMatrix{ElType}, stat_type::String="fz";
        cluster_sim_threshold::AbstractFloat=0.8, parallel="single",
    ordering="size", sim_mat::Matrix{Float64}=zeros(Float64, 0, 0), verbose::Bool=false, greedy::Bool=true,
    pw_uni_args::Dict{Symbol,T}=Dict{Symbol,Any}(), unsig_is_na::Bool=true)

    verbose && println("Computing pairwise similarities")

    if !greedy && unsig_is_na
        warn("Hierarchical clustering currently doesn't support NA values, assuming them to be max distance.")
        unsig_is_na = false
    end

    if isempty(sim_mat)
        sim_mat = pw_unistat_matrix(data, stat_type, parallel=parallel, pw_uni_args=pw_uni_args, unsig_is_na=unsig_is_na)
    end

    if stat_type == "mi"
        verbose && println("Computing entropies")

        # can potentially be improved by pre-allocation
        entrs = mapslices(x -> entropy(counts(x) ./ length(x)), data, 1)

    elseif stat_type == "mi_nz"
        nz_mask = data .!= 0

    elseif startswith(stat_type, "fz")
        sim_mat = abs.(sim_mat)
    end

    #greedy_mode = cluster_mode == "greedy"
    unclustered_vars = Set{Int}(1:size(data, 2))
    clust_dict = Dict{Int,Set{Int}}()

    if greedy
        var_order = collect(1:size(data, 2))
        if ordering == "size"
            var_sizes = sum(data, 1)
            sort!(var_order, by=x -> var_sizes[x], rev=true)
        end
    else
        var_order = 1:size(data, 2)
    end

    verbose && println("Clustering")

    if !greedy
        verbose && println("\tConverting similarities to normalized distances")
        dist_mat = similar(sim_mat)
    end

    n_obs_min::Int = haskey(pw_uni_args, :n_obs_min) ? pw_uni_args[:n_obs_min] : 1
    is_mi_stat = startswith(stat_type, "mi")
    is_fz_stat = startswith(stat_type, "fz")

    for var_A in var_order
        if var_A in unclustered_vars
            pop!(unclustered_vars, var_A)

            clust_members = Set(var_A)
            for var_B in unclustered_vars
                sim = sim_mat[var_A, var_B]

                if greedy && ((sim == 0.0) || isnan(sim))
                    continue
                end

                if is_mi_stat
                    if stat_type == "mi"
                        entr_A = entrs[var_A]
                        entr_B = entrs[var_B]
                    elseif stat_type == "mi_nz"
                        curr_nz_mask = (nz_mask[:, var_A] .& nz_mask[:, var_B])[:]
                        nz_elems = sum(curr_nz_mask)

                        if nz_elems < n_obs_min
                            entr_A = entr_B = norm_term = 0
                        else
                            entr_A = entropy(counts(data[curr_nz_mask, var_A]) ./ nz_elems)
                            entr_B = entropy(counts(data[curr_nz_mask, var_B]) ./ nz_elems)
                            norm_term = sqrt(entr_A * entr_B)
                        end
                    end
                    norm_term = sqrt(entr_A * entr_B)

                    sim = norm_term != 0.0 ? abs(sim) / norm_term : 0.0

                end

                if greedy
                    if sim > cluster_sim_threshold
                        push!(clust_members, var_B)
                        pop!(unclustered_vars, var_B)
                    end
                else
                    dist = 1.0 - sim
                    dist_mat[var_A, var_B] = dist
                    dist_mat[var_B, var_A] = dist
                end
            end

            if greedy
                clust_dict[var_A] = clust_members
            end
        end
    end

    if !greedy
        verbose && println("\tComputing hierarchical clusters")
        isdefined(:Clustering) || @eval using Clustering
        clust_dict = Base.invokelatest(cluster_hierarchical, dist_mat, data, cluster_sim_threshold)
    end

    (sort(collect(keys(clust_dict))), clust_dict)
end
