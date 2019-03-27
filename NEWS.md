# v0.14.0 (minor, latest)

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
