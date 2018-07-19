# FlashWeave

FlashWeave predicts ecological interactions between microbes from large-scale abundance data (i.e. OTU tables constructed from sequencing data) through statistical co-occurrence. It reports direct associations, corrected for bystander effects and other confounders, and can furthermore integrate environmental or technical factors into the analysis of microbial systems.


## Installation ##

To install Julia, please follow instructions on https://github.com/JuliaLang/julia. The prefered way is to obtain a binary from https://julialang.org/downloads/. Make sure you install Julia 0.6.x, the versions currently supported by FlashWeave.

In an interactive Julia session, you can then install FlashWeave via

```julia
Pkg.clone("https://github.com/meringlab/FlashWeave.jl")
```


## Basic usage ##

OTU tables can be provided in several formats, such as delimited formats (".csv", ".tsv"), Julia-specific JLD2 (".jld2") and BIOM (".biom"). Meta data should be provided as delimited format. IMPORTANT NOTE: FlashWeave treats rows of the table as observations (i.e. samples) and columns as variables (i.e. OTUs or meta variables) for delimited and JLD2 formats, consistent with the majority of statistical and machine-learning applications, but in contrast to several other microbiome analysis frameworks.

To learn an interaction network, you can do

```julia
julia> using FlashWeave # this has some pre-compilation delay the first time it's called, subsequent imports are fast

julia> data_path = "/my/example/data.tsv"
julia> meta_data_path = "/my/example/meta_data.tsv"
julia> netw_results = learn_network(data_path, meta_data=meta_data_path, sensitive=true, heterogeneous=false)

julia> # for JLD2, you can provide keys:
julia> # data_path = "/my/example/data.jld2"
julia> # netw_results = learn_network(data_path, data_key="otu_table", meta_key="meta_data_table", sensitive=true, heterogeneous=false)
```
Results can currently be saved in JLD (".jld2") or as edgelist (".edgelist") format:

```julia
julia> save_network("/my/example/network_output.jld2", netw_results)
julia> ## or: save_network("/my/example/network_output.edgelist", netw_results)
```
For output of additional information (if available), such as discarding sets, in separate files you can specify the "detailed" flag:

```julia
julia> save_network("/my/example/network_output.jld2", netw_results, detailed=true)
```

## Parallel computing ##

FlashWeave leverages Julia's built-in [parallel infrastructure](https://docs.julialang.org/en/stable/manual/parallel-computing/). In the most simple case, you can start julia with several workers

```bash
julia -p 4 # for 4 workers
```

or manually add workers at the beginning of an interactive session

```julia
julia> addprocs(4)
julia> using FlashWeave
julia> learn_network(...
```
and network learning will be parallelized in a shared-memory, multi-process fashion.

If you want to run FlashWeave remotely on a computing cluster, a ```ClusterManager``` can be used (for example from the [ClusterManagers.jl](https://github.com/JuliaParallel/ClusterManagers.jl) package). Details differ depending on the setup (queueing system, resource requirements, ...), but a simple example for a Sun Grid Engine (SGE) system would be:

```julia
julia> using ClusterManagers
julia> addprocs_qrsh(20) # 20 remote workers
julia> ## for more fine-grained control: addprocs(QRSHManager(20, "<your queue>"), qsub_env="<your environment>", params=Dict(:res_list=>"<requested resources>"))

julia> # or

julia> addprocs_sge(20)
julia> ## addprocs_sge(5, queue="<your queue>", qsub_env="<your environment>", res_list="<requested resources>")
```
Please refer to the [ClusterManagers.jl documentation](https://github.com/JuliaParallel/ClusterManagers.jl) for further details.
