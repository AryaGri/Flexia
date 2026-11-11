# Показ разреженного якобиана на кривошипе
# запуск: julia --project=. test/demo_jacobian.jl

using Flexia
using ForwardDiff
using SparseArrays

include("mechanisms.jl")

sys, initial = build_crankshaft()
n = number_of_dofs(sys)

println("размер системы n = ", n)
println("ненулевых элементов: ", nnz(sys.jac_pattern), " из ", n^2)
println("цветов: ", maximum(sys.jac_colors))
println("цвета столбцов: ", sys.jac_colors)

println("шаблон разреженности:")
display(sys.jac_pattern)
println()

# сравнение со старым плотным якобианом
J_new = sys.jacobian(initial)
J_old = ForwardDiff.jacobian(sys.rhs, initial)
println("тип нового якобиана: ", typeof(J_new))
println("тип старого якобиана: ", typeof(J_old))
println("максимальная разница: ", maximum(abs.(Matrix(J_new) - J_old)))
