# Dense ForwardDiff Jacobian vs sparse Jacobian, and full `simulate` time.
#
#   julia --project=. test/benchmark_jacobian.jl

using Flexia
using ForwardDiff
using LinearAlgebra
using Printf

include("mechanisms.jl")

# minimal time of `f()` over `repeats` runs after a warm-up call
function best_time(f; repeats=5)
    f()
    return minimum(@elapsed(f()) for _ in 1:repeats)
end

function jacobian_times(sys, state)
    dense = best_time(() -> ForwardDiff.jacobian(sys.rhs, state); repeats=50)
    sparse = best_time(() -> sys.jacobian(state); repeats=50)
    return dense, sparse
end

function simulate_times(build, time_span)
    sys_dense, initial = build(; sparse_jacobian=false)
    sys_sparse, _ = build()
    dense = best_time(() -> simulate(sys_dense, initial, time_span); repeats=3)
    times = Dict{Symbol,Float64}()
    for linear_solver in (:auto, :dense, :sparse)
        times[linear_solver] = best_time(() -> simulate(sys_sparse, initial, time_span; linear_solver); repeats=3)
    end
    return dense, times
end

cases = [
    ("crankshaft", (; kw...) -> build_crankshaft(; kw...)),
    ("linear actuator", (; kw...) -> build_linear_actuator(; kw...)),
    [("chain $n", (; kw...) -> build_chain(n; kw...)) for n in (2, 5, 10, 20, 50, 100)]...,
]

println("Jacobian of rhs, one evaluation")
@printf("%-16s %5s %6s %7s %12s %12s %8s\n", "mechanism", "n", "nnz", "colors", "dense, μs", "sparse, μs", "speedup")
for (name, build) in cases
    sys, initial = build()
    n = number_of_dofs(sys)
    cache = sys.sparse_jacobian
    dense, sparse = jacobian_times(sys, initial)
    @printf("%-16s %5d %6d %7d %12.1f %12.1f %7.1fx\n", name, n, length(cache.pattern.nzval),
        cache.ncolors, 1e6 * dense, 1e6 * sparse, dense / sparse)
end

println()
println("simulate, 200 steps")
@printf("%-16s %5s %12s %12s %12s %12s %8s\n", "mechanism", "n", "old, ms", "auto, ms", "dense LU, ms", "sparse LU, ms", "speedup")
for (name, build) in cases
    time_span = 0:0.001:0.2
    sys, _ = build()
    dense, times = simulate_times(build, time_span)
    @printf("%-16s %5d %12.2f %12.2f %12.2f %12.2f %7.1fx\n", name, number_of_dofs(sys), 1e3 * dense,
        1e3 * times[:auto], 1e3 * times[:dense], 1e3 * times[:sparse], dense / times[:auto])
end
