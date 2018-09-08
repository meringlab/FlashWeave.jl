start_time = time()

using FlashWeave, Distributed, Test

println("Preparing tests")

nprocs() == 1 && addprocs(1)

@testset "Total" begin
    for test_module in sort(filter(x -> endswith(x, ".jl") && x != "runtests.jl", readdir(pwd())))
        println("\nTesting $test_module")
        if test_module == "learning.jl"
            println("(this can take a couple of minutes)")
        end
        include(test_module)
    end
end

time_total = Int(round(time() - start_time))
println("\n\nFinished testing (took $(time_total)s)")
