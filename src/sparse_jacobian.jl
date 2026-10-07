# Sparse Jacobian of the system right-hand side.
#
# The structure of ∂rhs/∂state is fixed by the topology of the mechanism
# (which bodies are connected by which joints), so it is detected once in
# `assemble!` and reused on every time step:
#
#   1. Sparsity pattern — `sys.rhs` is evaluated once on symbolic tracer
#      numbers (SparseConnectivityTracer). Each tracer carries the set of
#      state entries it depends on, so the result is the exact structural
#      pattern for any state, not just for the one used for tracing.
#   2. Column coloring — columns that never share a row get the same color
#      (greedy distance-2 coloring). All columns of one color can be
#      differentiated with a single dual-number direction.
#   3. Compressed ForwardDiff — `sys.rhs` is evaluated with `ncolors`
#      partials instead of `n`, and the values are scattered straight into
#      the nonzeros of a `SparseMatrixCSC`.
#
# The cost of one Jacobian is therefore ⌈ncolors / 12⌉ evaluations of rhs
# instead of ⌈n / 12⌉, and ncolors does not grow with the number of bodies.

struct JacobianTag end

# ForwardDiff's default maximal chunk size: never use more partials per pass
# than the dense `ForwardDiff.jacobian` did.
const MAX_JACOBIAN_CHUNK = 12

mutable struct SparseJacobianCache{N}
    pattern::SparseMatrixCSC{Float64,Int64}
    colors::Vector{Int64}
    ncolors::Int64
    xdual::Vector{ForwardDiff.Dual{JacobianTag,Float64,N}}
    # `sys.jacobian` created by `assemble!`. If a user replaces
    # `sys.jacobian`, the solver falls back to the dense path with it.
    jacobian_fn::Function
    # false when the pattern could not be traced and a dense one is used
    traced::Bool
end

number_of_colors(cache::SparseJacobianCache) = cache.ncolors
jacobian_pattern(cache::SparseJacobianCache) = cache.pattern

# Records which entries `update_jacobian!` of connectors writes to, so that
# those entries can be added to the sparsity pattern.
struct EntryRecorder <: AbstractMatrix{Float64}
    m::Int64
    n::Int64
    rows::Vector{Int64}
    cols::Vector{Int64}
end
EntryRecorder(m, n) = EntryRecorder(m, n, Int64[], Int64[])
Base.size(r::EntryRecorder) = (r.m, r.n)
Base.getindex(::EntryRecorder, i::Int, j::Int) = 0.0
function Base.setindex!(r::EntryRecorder, v, i::Int, j::Int)
    push!(r.rows, i)
    push!(r.cols, j)
    return v
end

function detect_rhs_sparsity(rhs::Function, state::Vector{Float64})
    try
        return jacobian_sparsity(rhs, state, TracerSparsityDetector()), true
    catch err
        # Tracers have no numeric value, so rhs code with value-dependent
        # branches (`if t < 1 ...`) cannot be traced. A dense pattern keeps
        # the result correct and the cost equal to the dense ForwardDiff.
        @warn "Could not detect the sparsity pattern of the rhs, the Jacobian will be computed as dense" exception = (err, catch_backtrace())
        n = length(state)
        return sparse(trues(n, n)), false
    end
end

# Greedy distance-2 column coloring, largest columns first: two columns get
# different colors if they have a nonzero in the same row.
function color_columns(pattern::SparseMatrixCSC)
    n = size(pattern, 2)
    rows = rowvals(pattern)
    by_row = sparse(pattern')
    cols_in_row = rowvals(by_row)

    colors = zeros(Int64, n)
    forbidden = zeros(Int64, n + 1)
    order = sortperm([length(nzrange(pattern, j)) for j in 1:n]; rev=true)
    for j in order
        for idx in nzrange(pattern, j), idx2 in nzrange(by_row, rows[idx])
            c = colors[cols_in_row[idx2]]
            c > 0 && (forbidden[c] = j)
        end
        c = 1
        while forbidden[c] == j
            c += 1
        end
        colors[j] = c
    end
    return colors
end

function SparseJacobianCache(sys, state::Vector{Float64})
    n = length(state)
    traced_pattern, traced = detect_rhs_sparsity(sys.rhs, state)

    recorder = EntryRecorder(n, n)
    update_act_jacobian!(sys, recorder)

    rows, cols, _ = findnz(traced_pattern)
    append!(rows, recorder.rows)
    append!(cols, recorder.cols)
    pattern = sparse(rows, cols, zeros(length(rows)), n, n, (a, b) -> a)

    colors = color_columns(pattern)
    ncolors = n == 0 ? 0 : maximum(colors)
    # balance the passes: 13 colors -> 2 passes of 7, not 12 + 1
    npasses = max(1, cld(ncolors, MAX_JACOBIAN_CHUNK))
    N = max(1, cld(ncolors, npasses))
    xdual = Vector{ForwardDiff.Dual{JacobianTag,Float64,N}}(undef, n)
    return SparseJacobianCache{N}(pattern, colors, ncolors, xdual, identity, traced)
end

@inline function seed(::Val{N}, k::Int) where {N}
    return ForwardDiff.Partials{N,Float64}(ntuple(i -> Float64(i == k), Val(N)))
end

"""
    sparse_jacobian!(J, rhs, state, cache)

Fill the nonzeros of `J` (which must have the structure of
`cache.pattern`) with ∂rhs/∂state at `state`.
"""
function sparse_jacobian!(
    J::SparseMatrixCSC{Float64,Int64}, rhs::Function,
    state::AbstractVector{Float64}, cache::SparseJacobianCache{N}) where {N}

    colors = cache.colors
    xdual = cache.xdual
    rows = rowvals(J)
    vals = nonzeros(J)
    for first_color in 1:N:cache.ncolors
        for j in eachindex(xdual)
            k = colors[j] - first_color + 1
            xdual[j] = ForwardDiff.Dual{JacobianTag}(state[j], seed(Val(N), k))
        end
        ydual = rhs(xdual)
        for j in axes(J, 2)
            k = colors[j] - first_color + 1
            1 <= k <= N || continue
            for idx in nzrange(J, j)
                vals[idx] = ForwardDiff.partials(ydual[rows[idx]], k)
            end
        end
    end
    return J
end
