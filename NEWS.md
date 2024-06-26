# master (unpublished)

### general

- strongly speed-up contingency table computation for heterogeneous=true and max_k=0/1
- use the more compiler-friendly `stack` (introduced in Julia v1.9) instead of `hcat` for large numbers of columns (if available), introduce fast method for sparse columns
- improve univariate pvalue filtering
- remove performance bottleneck in three-way `adjust_df`
- catch incorrect usage of 'meta_data_path' as keyword argument

# v0.19.2 (latest)

### bug fixes

- force `si_HITON_PC` to remove non-significant univariate candidates earlier
- fix edge case where direct usage of the unexposed sparse processing engine could lead to infinite loops
- fix SparseArrays incompatibility (surfaced by julia v1.9)

# v0.19.1

### general

- expose experimental keyword arguments within `learn_network` for easier testing (subject to change, syntax/functionality should not be relied upon)

### bug fixes

- fix dependency issue with HDF5.jl@v0.16

# v0.19.0

### general

- add support for shared-memory OTU tables (sparse and dense), invoked via the `share_data` flag in `learn_network`
- improve speed and edge case behaviour for non-recursive partial correlations
- improve freeing of memory (in particular on parallel workers) between major network inference steps
- improve error reporting on parallel workers

### bug fixes

- fix numerical edge case for recursive partial correlations
- fix handling of variables with no absences when `heterogeneous=true`

# v0.18.1

### bug fixes

- add hotfix for irregular breaking changes in SimpleWeightedGraphs.jl
- fix bug that prevented adaptive_clr from correctly propagating samples that were dropped due to numerical edge cases (thanks **@jonasjonker** for reporting)

# v0.18.0

### general

- `fast_elim` flag now also exposed via `learn_network`
- fixed documentation for `save_network`/`load_network` to drop outdated .tsv/.csv mention (thanks **@ARW-UBT** for reporting)
- update dependency versions

### bug fixes

- fix bug that prevented combining sparse .biom data with non-numeric meta data (issue #20, thanks **@ARW-UBT** for reporting)

# v0.17.0

### general
- Julia versions < 1.2 are no longer supported; this change was necessary for the interleaved parallelism
overhaul in this version
- the interleaved parallelism backend got fixed to properly support changes introduced in Julia 1.2;
it now runs more stably and features better error reporting

### bug fixes
- fixed occasional hangs when computing networks in parallel via the interleaved mode (issue #9)

# v0.16.0

### general
- add explicit parallelism flag (`parallel_mode`) to `learn_network` to avoid having to remove workers when switching from multi-core to single-core computations
- improve error messages in several places

### bug fixes
- fix edge case in bin filtering: variables w/o zero entries are now retained and properly discretized
- ensure that `meta_mask` is a proper `BitVector` in `learn_network` and `normalize_data`

# v0.15.0

### general
- improve sign determination for conditional mutual information tests
- remove jld2 support due to stability issues
- categorical meta data elements must now include non-numeric characters to ensure
dintinguishability from continuous columns

### bug fixes
- fix bug in `read_edgelist` where files with unconnected nodes at high
header positions could not be read
- fix bug where continuous variables were sometimes one-hot encoded (thanks **@pbelmann** for reporting)
- fix bug where zero weights were assigned to highly associated variable pairs in rare cases (thanks **@Mortomoto** for reporting)
- fix bug that could lead to the first data column being parsed as row identifiers if it had no duplicate entries
- fix `save_network` with `detailed=true` not producing output in the latest version (thanks **@pbelmann** for reporting)

# v0.14.0

### general
- allow string factors in meta data
- make one-hot encoding the default for meta data variables with more than two
categories, this avoids unreliable tests and improves interpretability (can be switched off with "make_onehot = false")
- improve support for continuous meta data variables when using "fast=true"
- support numeric OTU identifiers

### bug fixes
- fix Travis CI osx handling

## v0.13.1
- fix stdlib dependency issue

# v0.13.0

### general
- make Travis CI handle osx (fix HDF5 compilation)
- remove several unmaintained backend options (temporary output, AND rule, pvalue weights)
- make adaptive CLR normalized matrices always dense to avoid inefficiencies
- drop samples with adaptive pseudo-counts below machine precision for now

### bug fixes
- fix meta_mask bug in data normalization
- fix performance regression in CLR transform of sparse matrices
