start_time = time()

println("Preparing tests")

if nprocs() == 1
    addprocs(1)
end

using Cauocc
using Base.Test

for test_module in ["preprocessing.jl", "misc.jl", "contingency.jl", "statfuns.jl",
                    "tests.jl", "learning.jl"]

    println("\nTesting $test_module")
    if test_module == "learning.jl"
        println("(this will take a moment)")
    end
    include(test_module)
end

time_total = Int(round(time() - start_time))
println("\n\n Finished testing (took $(time_total) s))")
