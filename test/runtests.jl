using Flexia
using Test
using ForwardDiff
using SparseArrays
using Random

include("mechanisms.jl")

@testset "Flexia.jl" begin
    @test test_func() == 1
end

stored_entries(P::SparseMatrixCSC) = Set(zip(findnz(P)[1], findnz(P)[2]))

@testset "sparse jacobian" begin
    rng = MersenneTwister(1)
    cases = [
        ("crankshaft", () -> build_crankshaft()),
        ("chain of 10", () -> build_chain(10)),
        ("linear actuator", () -> build_linear_actuator()),
    ]
    for (name, build) in cases
        @testset "$name" begin
            sys, initial = build()
            cache = sys.sparse_jacobian
            @test cache !== nothing
            @test cache.traced

            n = number_of_dofs(sys)
            J = sys.jacobian(initial)
            @test J isa SparseMatrixCSC{Float64,Int64}
            @test nnz(J) < n^2 ÷ 2
            @test Flexia.number_of_colors(cache) < n

            # exact match with the dense ForwardDiff Jacobian in random states
            for _ in 1:5
                state = randn(rng, n)
                @test Matrix(sys.jacobian(state)) ≈ ForwardDiff.jacobian(sys.rhs, state) rtol = 1e-12 atol = 1e-12
            end
        end
    end

    @testset "coloring is valid" begin
        sys, _ = build_chain(10)
        cache = sys.sparse_jacobian
        P = cache.pattern
        rows = rowvals(P)
        for i in 1:size(P, 1)
            cols = [j for j in 1:size(P, 2) if i in rows[nzrange(P, j)]]
            @test allunique(cache.colors[cols])
        end
    end

    @testset "actuator entry is in the pattern" begin
        sys, _ = build_linear_actuator()
        P = sys.sparse_jacobian.pattern
        actuator = only(c for c in connectors(sys) if c isa PositionLinearActuator2D)
        lm = Flexia.get_lms(sys, actuator)[1]
        @test (lm, size(P, 2)) in stored_entries(P)
    end

    @testset "untraceable rhs falls back to dense" begin
        sys = MBSystem2D()
        bd = Body2D(1.0, 1.0)
        bd.forces[1] = (x, t) -> t < 1 ? 1.0 : 0.0
        add!(sys, bd)
        @test_logs (:warn, r"sparsity pattern") match_mode = :any assemble!(sys)
        @test !sys.sparse_jacobian.traced
        state = randn(rng, number_of_dofs(sys))
        @test Matrix(sys.jacobian(state)) ≈ ForwardDiff.jacobian(sys.rhs, state)
    end
end

@testset "simulate: sparse == dense" begin
    cases = [
        ((; kw...) -> build_crankshaft(; kw...), 0:0.01:1.0),
        ((; kw...) -> build_chain(10; kw...), 0:0.01:1.0),
        ((; kw...) -> build_linear_actuator(; kw...), 0:0.001:0.5),
    ]
    for (build, time_span) in cases
        sys_sparse, initial = build()
        sys_dense, _ = build(; sparse_jacobian=false)
        reference = simulate(sys_dense, initial, time_span)
        # the dense LU sees exactly the same matrix as before
        @test simulate(sys_sparse, initial, time_span; linear_solver=:dense) == reference
        # UMFPACK pivots in another order; the crankshaft (ground mass 1e6)
        # is ill-conditioned, so only round-off level agreement is expected
        @test simulate(sys_sparse, initial, time_span; linear_solver=:sparse) ≈ reference rtol = 1e-7
    end
end
