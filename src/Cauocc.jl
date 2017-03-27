__precompile__()

module Cauocc

include("statfuns.jl")
include("misc.jl")
include("contingency.jl")
include("tests.jl")
include("stackchannels.jl")
include("learning.jl")
include("preprocessing.jl")

using Cauocc.Learning

export LGL, si_HITON_PC

#precompile(LGL, (Matrix{Int64}, ))
#precompile(si_HITON_PC, (Int64, Matrix{Int64}, ))

#precompile(LGL, (Matrix{Int64}; test_name="mi")
"""
# precompile hints
data = readdlm("/mnt/mnemo3/janko/.julia/v0.5/v0.5/Cauocc/data/precomp_data.tsv", '\t')
data_sign = Cauocc.Preprocessing.preprocess_data(data, "binary", verbose=false)
data_sign_sp = sparse(data_sign)
LGL(data_sign; test_name="mi", max_k=3, verbose=false)
LGL(data_sign_sp; test_name="mi", max_k=3, verbose=false)

data_tri = Cauocc.Preprocessing.preprocess_data(data, "binned_nz", verbose=false)
data_tri_sp = sparse(data_tri)
LGL(data_tri; test_name="mi_nz", max_k=3, verbose=false)
LGL(data_tri_sp; test_name="mi_nz", max_k=3, verbose=false)

data_rows = Cauocc.Preprocessing.preprocess_data(data, "rows", verbose=false)
data_rows_sp = sparse(data_rows)
LGL(data_rows; test_name="fz", max_k=3, verbose=false)
LGL(data_rows_sp; test_name="fz", max_k=3, verbose=false)
"""
end