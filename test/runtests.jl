if nprocs() == 1
    addprocs(1)
end

using Cauocc
using Base.Test

for test_module in ["preprocessing.jl", "misc.jl", "contingency.jl", "statfuns.jl",
                    "tests.jl", "learning.jl"]
    println("\n\nTesting $test_module")
    include(test_module)
end
#include("preprocessing.jl")
#include("misc.jl")
#include("contingency.jl")
#include("statfuns.jl")
#include("tests.jl")
#include("learning.jl")
