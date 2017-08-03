start_time = time()

println("Preparing tests")

#if nprocs() == 1
#    addprocs(1)
#end
warn("Multicore currently disabled due to random crashes")

using FlashWeave
using Base.Test

for test_module in ["preprocessing.jl", "misc.jl", "contingency.jl", "statfuns.jl",
                    "tests.jl", "learning.jl"]

    println("\nTesting $test_module")
    if test_module == "learning.jl"
        println("(this can take a couple of minutes)")
    end
    include(test_module)
end

time_total = Int(round(time() - start_time))
println("\n\n Finished testing (took $(time_total)s)")
