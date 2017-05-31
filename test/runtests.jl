if nprocs() == 1
    addprocs(1)
end

using Cauocc
using Base.Test

for test_module in ["preprocessing.jl", "misc.jl", "contingency.jl", "statfuns.jl",
                    "tests.jl", "learning.jl"]
    println("\nTesting $test_module")
    include(test_module)
end
