# Сравнение скорости: старый плотный якобиан и новый разреженный
# запуск: julia --project=. test/benchmark_jacobian.jl

using Flexia
using ForwardDiff

include("mechanisms.jl")

# минимальное время из нескольких запусков (первый запуск не считаем, там компиляция)
function min_time(f, repeats)
    f()
    best = Inf
    for i in 1:repeats
        best = min(best, @elapsed f())
    end
    return best
end

builders = [
    ("crankshaft", build_crankshaft),
    ("linear actuator", build_linear_actuator),
    ("chain 2", (; kw...) -> build_chain(2; kw...)),
    ("chain 5", (; kw...) -> build_chain(5; kw...)),
    ("chain 10", (; kw...) -> build_chain(10; kw...)),
    ("chain 20", (; kw...) -> build_chain(20; kw...)),
    ("chain 50", (; kw...) -> build_chain(50; kw...)),
]

time_span = 0:0.001:0.2

println("name | n | colors | jac old, mks | jac new, mks | simulate old, ms | simulate new, ms")
for (name, build) in builders
    sys, initial = build()
    sys_old, _ = build(sparse_jacobian=false)
    n = number_of_dofs(sys)

    t_jac_old = min_time(() -> sys_old.jacobian(initial), 30)
    t_jac_new = min_time(() -> sys.jacobian(initial), 30)
    t_sim_old = min_time(() -> simulate(sys_old, initial, time_span), 3)
    t_sim_new = min_time(() -> simulate(sys, initial, time_span), 3)

    println(name, " | ", n, " | ", maximum(sys.jac_colors), " | ",
        round(t_jac_old * 1e6, digits=1), " | ", round(t_jac_new * 1e6, digits=1), " | ",
        round(t_sim_old * 1e3, digits=2), " | ", round(t_sim_new * 1e3, digits=2))
end
