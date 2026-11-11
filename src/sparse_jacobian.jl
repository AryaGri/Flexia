# Разреженный якобиан правой части.
#
# Идея: какие элементы якобиана ненулевые, зависит только от того, какие тела
# соединены какими шарнирами. Поэтому шаблон ищем один раз в assemble!,
# а потом на каждом шаге считаем только ненулевые элементы.

# Шаблон разреженности. В rhs вместо чисел подставляются специальные "числа"
# из SparseConnectivityTracer, которые запоминают, от каких переменных они зависят.
function find_jacobian_pattern(sys)
    n = number_of_dofs(sys)
    pattern = spzeros(Bool, n, n)
    try
        pattern = jacobian_sparsity(sys.rhs, zeros(n), TracerSparsityDetector())
    catch
        # так бывает, если в силе есть if по времени или по состоянию
        @warn "Не получилось найти шаблон разреженности, якобиан будет плотным"
        pattern = sparse(ones(Bool, n, n))
    end

    # актуаторы в update_jacobian! пишут производную по времени в последний столбец,
    # в rhs этой зависимости нет, поэтому добавляем эти элементы руками
    for connector in connectors(sys)
        if connector isa AbstractPositionActuator2D
            lm = get_lms(sys, connector)[1]
            pattern[lm, n] = true
        end
    end
    return Float64.(pattern)
end

# Раскраска столбцов: два столбца можно покрасить в один цвет,
# если у них нет ненулевых элементов в одной и той же строке
function color_columns(pattern)
    n = size(pattern, 2)
    rows = rowvals(pattern)
    colors = zeros(Int64, n)
    # для каждой строки храним цвета, которые в ней уже заняты
    used_colors = [Set{Int64}() for i in 1:size(pattern, 1)]
    for j in 1:n
        col_rows = rows[nzrange(pattern, j)]
        c = 1
        while any(c in used_colors[i] for i in col_rows)
            c += 1
        end
        colors[j] = c
        for i in col_rows
            push!(used_colors[i], c)
        end
    end
    return colors
end

function calc_sparse_jacobian(sys, state)
    pattern = sys.jac_pattern
    colors = sys.jac_colors
    ncolors = maximum(colors)

    # все переменные одного цвета сдвигаем на одно и то же c[цвет],
    # тогда производных нужно ncolors штук, а не n
    shifted_rhs = (c) -> sys.rhs([state[j] + c[colors[j]] for j in eachindex(state)])
    B = ForwardDiff.jacobian(shifted_rhs, zeros(ncolors))

    # раскладываем сжатый якобиан обратно по своим местам
    J = copy(pattern)
    rows = rowvals(J)
    vals = nonzeros(J)
    for j in 1:size(J, 2)
        for k in nzrange(J, j)
            vals[k] = B[rows[k], colors[j]]
        end
    end
    return J
end
