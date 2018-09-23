start_time = time()

using FlashWeave, Distributed, Test

println("Preparing tests")

nprocs() == 1 && addprocs(1)

test_modules = ["io.jl", "preprocessing.jl", "contingency.jl", "statfuns.jl", "tests.jl",
                "misc.jl", "learning.jl"]

@testset "all modules in testset" begin
    @test Set(test_modules) == Set(filter(x -> endswith(x, ".jl") && x != "runtests.jl", readdir(pwd())))
end

for test_module in test_modules
    println("\nTesting $test_module")
    if test_module == "learning.jl"
        println("(this can take a couple of minutes)")
    end
    include(test_module)
end

time_total = Int(round(time() - start_time))
println("\n\nFinished testing (took $(time_total)s)")
