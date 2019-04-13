# FlashWeave #

[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Build Status](https://travis-ci.org/meringlab/FlashWeave.jl.svg?branch=master)](https://travis-ci.org/meringlab/FlashWeave.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/vdesge86ssj91htc?svg=true)](https://ci.appveyor.com/project/jtackm/flashweave-jl)
[![codecov](https://codecov.io/gh/meringlab/FlashWeave.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/meringlab/FlashWeave.jl)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

FlashWeave predicts ecological interactions between microbes from large-scale compositional abundance data (i.e. OTU tables constructed from sequencing data) through statistical co-occurrence or co-abundance. It reports direct associations, with adjustment for bystander effects and other confounders, and can furthermore integrate environmental or technical factors into the analysis of microbial systems.

## Installation ##

To install Julia, please follow instructions on https://github.com/JuliaLang/julia. The preferred way is to obtain a binary from https://julialang.org/downloads/. Make sure you install Julia 1.0 or above, the versions currently supported by FlashWeave.

In an interactive Julia session, you can then install FlashWeave after typing `]` via

```julia
(v1.0) pkg> add FlashWeave
# to run tests: (v1.0) pkg> test FlashWeave
```

## Basic usage ##

To learn an interaction network from an OTU table and (optionally) a meta data table, you can do

```julia
julia> using FlashWeave # this has some pre-compilation delay the first time it's called, subsequent imports are fast

julia> data_path = "/my/example/data.tsv" # or .csv, .biom
julia> meta_data_path = "/my/example/meta_data.tsv"
julia> netw_results = learn_network(data_path, meta_data_path, sensitive=true, heterogeneous=false)

<< summary statistics of the learned network >>

julia> G = graph(netw_results) # weighted graph representing interactions + weights
```

Results can currently be saved in JLD2, fast for large networks, or as traditional [Graph Modelling Language](https://en.wikipedia.org/wiki/Graph_Modelling_Language) (".gml") or edgelist (".edgelist") formats:

```julia
julia> save_network("/my/example/network_output.jld2", netw_results)
julia> ## or: save_network("/my/example/network_output.gml", netw_results)
```

For output of additional information (such as discarding sets, if available) in separate files you can specify the "detailed" flag:

```julia
julia> save_network("/my/example/network_output.edgelist", netw_results, detailed=true)
julia> # for .jld2, additional information is always saved if available
```

A convenient loading function is available:
 ```julia
julia> netw_results = load_network("/my/example/network_output.jld2")
 ```

To get more information on a function, you may type `?` into the prompt, followed by a function name:

```
julia> ?
help> learn_network

    learn_network(data::AbstractArray{<:Real}) -> FWResult{Int}

Learn an interaction network from a data table (including OTUs and optionally meta variables).

   Algorithmic parameters:

    •    heterogeneous - enable heterogeneous mode for multi-habitat or -protocol data with at least thousands of samples (FlashWeaveHE)

    •    sensitive - enable fine-grained associations (FlashWeave-S, FlashWeaveHE-S), sensitive=false results in the fast modes FlashWeave-F or FlashWeaveHE-F

    •    max_k - maximum size of conditioning sets, high values can strongly increase runtime. max_k=0 results in no conditioning (univariate mode)

    •    alpha - threshold used to determine statistical significance

    •    conv - convergence threshold, i.e. if conv=0.01 assume convergence if the number of edges increased by only 1% after 100% more runtime (checked in
        intervals)

    •    feed_forward - enable feed-forward heuristic

    •    max_tests - maximum number of conditional tests that should be performed on a variable pair before association is assumed

    •    hps - reliability criterion for statistical tests when sensitive=false

    •    FDR - perform False Discovery Rate correction (Benjamini-Hochberg method) on pairwise associations

    •    n_obs_min - don't compute associations between variables having less reliable samples (i.e. non-zero if heterogeneous=true) than this number. -1:
        automatically choose a threshold.

    •    time_limit - if feed-forward heuristic is active, determines the interval (seconds) at which neighborhood information is updated

  General parameters:

    •    normalize - automatically choose and perform data normalization (based on sensitive and heterogeneous)

    •    track_rejections - store for each discarded edge, which variable set lead to its exclusion (can be memory intense for large networks)

    •    verbose - print progress information

    •    transposed - if true, rows of data are variables and columns are samples

    •    prec - precision in bits to use for calculations (16, 32, 64 or 128)

    •    make_sparse - use a sparse data representation (should be left at true in almost all cases)

    •    update_interval - if verbose=true, determines the interval (seconds) at which network stat updates are printed
```

## Input data formats ##

### OTU tables ###

OTU tables can be provided in several formats: 

**delimited formats**: ".tsv" ([example](https://github.com/meringlab/FlashWeave.jl/tree/master/test/data/HMP_SRA_gut/HMP_SRA_gut_tiny.tsv)) or ".csv" ([example](https://github.com/meringlab/FlashWeave.jl/tree/master/test/data/HMP_SRA_gut/HMP_SRA_gut_tiny.csv))

**BIOM**: BIOM 1.0 ([description](http://biom-format.org/documentation/format_versions/biom-1.0.html), [example](https://github.com/meringlab/FlashWeave.jl/tree/master/test/data/HMP_SRA_gut/HMP_SRA_gut_tiny_json.biom)) or the more performant
BIOM 2.0 ([description](http://biom-format.org/documentation/format_versions/biom-2.0.html), [example](https://github.com/meringlab/FlashWeave.jl/tree/master/test/data/HMP_SRA_gut/HMP_SRA_gut_tiny_hdf5.biom))

**JLD2**: a julia-specific, high-performance file format ([description](https://github.com/simonster/JLD2.jl), [example](https://github.com/meringlab/FlashWeave.jl/tree/master/test/data/HMP_SRA_gut/HMP_SRA_gut_tiny_plus_meta.jld2))

### Meta data tables ###

Meta data should generally be provided as delimited format (see for instance [example1](https://github.com/meringlab/FlashWeave.jl/blob/master/test/data/HMP_SRA_gut/HMP_SRA_gut_tiny_meta.tsv) or [example2](https://github.com/meringlab/FlashWeave.jl/blob/master/test/data/HMP_SRA_gut/HMP_SRA_gut_tiny_meta_oneHotTest.tsv)), separately from the OTU table. Notably, this implies that FlashWeave does not yet support reading meta data directly from BIOM files, but requires a separate delimited meta data file (support will be added in an upcoming version).

For JLD2, however, you can already provide HDF5 keys linked to meta data tables (and optionally headers):

```julia
julia> data_path = "/my/example/otu_and_meta_data.jld2"
julia> netw_results = learn_network(data_path, otu_data_key="otu_data", otu_header_key="otu_header", meta_data_key="meta_data", meta_header_key="meta_header", sensitive=true, heterogeneous=false)
```

See also the [test/data/HMP_SRA_gut](https://github.com/meringlab/FlashWeave.jl/tree/master/test/data/HMP_SRA_gut) directory for further examples of OTU and meta data tables.

### Important notes ###

#### Samples as columns ####
For delimited and JLD2 formats, FlashWeave treats rows of the table as observations (i.e. samples) and columns as variables (i.e. OTUs or meta variables), consistent with the majority of statistical and machine-learning applications, but in contrast to several other microbiome analysis frameworks. Behavior can be switched with the ```transposed=true``` flag.

#### One-hot encoding of meta variables ####
Meta variables with more than two categories are automatically one-hot encoded by FlashWeave prior to network inference to increase the reliability and interpretability of statistical tests (the user will be notified if this happens). For instance, the meta variable

| HABITAT       |
| ------------- |
| soil          |
| soil          |
| marine        |
| river         |
| marine        |

will be split into three dummy variables in the following fashion

| HABITAT_soil  | HABITAT_marine  | HABITAT_river  |
| ------------- |:---------------:| --------------:|
| 1             | 0		  | 0		   |
| 1             | 0		  | 0		   |
| 0             | 1		  | 0		   |
| 0             | 0		  | 1		   |
| 0             | 1		  | 0		   |

Each dummy variable will be a separate node in the result network.

#### Treatment of missing values ####

FlashWeave currently does not support missing data, please remove all samples with missing entries (both in OTU and meta data tables) prior to running FlashWeave.

## Performance tips ##

Depending on your data, make sure to chose the appropriate flags (```heterogeneous=true``` for multi-habitat or -protocol data sets with ideally at least thousands of samples; ```sensitive=false``` for faster, but more coarse-grained associations) to achieve optimal runtime. If FlashWeave should get stuck on a small fraction of nodes with large neighborhoods, try increasing the convergence criterion (```conv```). To learn a network in parallel, see the section below.

Note, that this package is optimized for large-scale data sets. On small data (hundreds of samples and OTUs) its speed advantages can be negated by JIT-compilation overhead.

## Parallel computing ##

FlashWeave leverages Julia's built-in [parallel infrastructure](https://docs.julialang.org/en/v1/manual/parallel-computing/index.html). In the most simple case, you can start julia with several workers

```bash
$ julia -p 4 # for 4 workers
```

or manually add workers at the beginning of an interactive session

```julia
julia> using Distributed; addprocs(4) # can be skipped if julia was started with -p
julia> @everywhere using FlashWeave
julia> learn_network(...
```

and network learning will be parallelized in a shared-memory, multi-process fashion.

If you want to run FlashWeave remotely on a computing cluster, a ```ClusterManager``` can be used (for example from the [ClusterManagers.jl](https://github.com/JuliaParallel/ClusterManagers.jl) package, installable via ```]``` and then ```add ClusterManagers```). Details differ depending on the setup (queueing system, resource requirements etc.), but a simple example for a Sun Grid Engine (SGE) system would be:

```julia
julia> using ClusterManagers
julia> addprocs_qrsh(20) # 20 remote workers
julia> ## for more fine-grained control: addprocs(QRSHManager(20, "<your queue>"), qsub_env="<your environment>", params=Dict(:res_list=>"<requested resources>"))

julia> # or

julia> addprocs_sge(20)
julia> ## addprocs_sge(5, queue="<your queue>", qsub_env="<your environment>", res_list="<requested resources>")
```

Please refer to the [ClusterManagers.jl documentation](https://github.com/JuliaParallel/ClusterManagers.jl) for further details.

## Citing ##

To cite FlashWeave, please refer to our [preprint on bioRxiv](https://www.biorxiv.org/content/early/2018/08/13/390195):

```
Tackmann, Janko, Joao Frederico Matias Rodrigues, and Christian von Mering. "Rapid inference
of direct interactions in large-scale ecological networks from heterogeneous microbial
sequencing data." bioRxiv (2018): 390195.
```

Example BibTeX entry:

```
@article {tackmann2018rapid,
	author = {Tackmann, Janko and Rodrigues, Joao Frederico Matias and von Mering, Christian},
	title = {Rapid inference of direct interactions in large-scale ecological networks from heterogeneous microbial sequencing data},
	year = {2018},
	doi = {10.1101/390195},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2018/08/13/390195},
	eprint = {https://www.biorxiv.org/content/early/2018/08/13/390195.full.pdf},
	journal = {bioRxiv}
}
```

## Versioning and API ##

FlashWeave follows [semantic versioning](https://semver.org/). Stability guarantees are only provided for exported functions (official API), anything else should be considered untested and subject to change. Note, that FlashWeave is currently in its experimental phase (version < v1.0), which means that breaking interface changes may occur in every minor version.
