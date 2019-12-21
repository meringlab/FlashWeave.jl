# master (unpublished)

No changes yet

# v0.15.0 (latest)

### general
- improve sign determination for conditional mutual information tests
- remove jld2 support due to stability issues
- categorical meta data elements must now include non-numeric characters to ensure
dintinguishability from continuous columns

### bug fixes
- fix bug in read_edgelist where files with unconnected nodes at high
header positions could not be read
- fix bug where continuous variables were sometimes one-hot encoded
- fix bug where zero weights were assigned to highly associated variable pairs in rare cases
- fix bug that could lead to the first data column being parsed as row identifiers if it had no duplicate entries

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
