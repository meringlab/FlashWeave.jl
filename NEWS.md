# Latest

## general

- make Travis CI handle osx (fix HDF5 compilation)
- remove several unmaintained backend options (temporary output, AND rule, pvalue weights)
- make adaptive CLR normalized matrices always dense to avoid inefficiencies
- drop samples with adaptive pseudo-counts below machine precision for now

## bug fixes
- fix meta_mask bug in data normalization
- fix performance regression in CLR transform of sparse matrices
