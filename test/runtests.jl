using Flexia
using Test
using ForwardDiff
using SparseArrays
using Random

include("mechanisms.jl")

@testset "Flexia.jl" begin
    @test test_func() == 1
end

@testset "sparse jacobian" begin
    Random.seed!(1)
    systems = [build_crankshaft(), build_chain(10), build_linear_actuator()]
    for (sys, initial) in systems
        n = number_of_dofs(sys)
        J = sys.jacobian(initial)
        @test J isa SparseMatrixCSC
        @test maximum(sys.jac_colors) < n

        # сравниваем с обычным плотным якобианом в случайных точках
        for k in 1:5
            state = randn(n)
            @test Matrix(sys.jacobian(state)) ≈ ForwardDiff.jacobian(sys.rhs, state)
        end
    end

    # если в силе есть if, шаблон найти нельзя, но ответ все равно должен быть правильный
    sys = MBSystem2D()
    bd = Body2D(1.0, 1.0)
    bd.forces[1] = (x, t) -> t < 1 ? 1.0 : 0.0
    add!(sys, bd)
    assemble!(sys)
    state = randn(number_of_dofs(sys))
    @test Matrix(sys.jacobian(state)) ≈ ForwardDiff.jacobian(sys.rhs, state)
end

@testset "simulate" begin
    # траектория с разреженным якобианом должна совпадать со старой
    time_span = 0:0.01:1.0

    sys, initial = build_crankshaft()
    sys_old, _ = build_crankshaft(sparse_jacobian=false)
    @test simulate(sys, initial, time_span) ≈ simulate(sys_old, initial, time_span)

    sys, initial = build_linear_actuator()
    sys_old, _ = build_linear_actuator(sparse_jacobian=false)
    @test simulate(sys, initial, time_span) ≈ simulate(sys_old, initial, time_span)

    # 10 звеньев - это уже n > 80, тут работает разреженное LU
    sys, initial = build_chain(10)
    sys_old, _ = build_chain(10, sparse_jacobian=false)
    @test simulate(sys, initial, time_span) ≈ simulate(sys_old, initial, time_span) rtol = 1e-6
end
