mutable struct MBSystem2D
    bodies::Vector{AbstractBody2D}
    # joints::Vector{AbstractJoint2D}
    connectors::Vector{AbstractConnector2D}
    forces::Vector{AbstractForce2D}
    sensors::Vector{AbstractSensor2D}
    bodiesdofs::Vector{Int64}
    lmdofs::Vector{Int64}
    assembled::Bool
    rhs::Function
    jacobian::Function
    prestep::Function
    measure!::Function
    kinematic_constrains!::Function
    targets::Vector{Float64}
    mass::Matrix{Float64}
    jac_pattern::SparseMatrixCSC{Float64, Int64}  # шаблон разреженности якобиана
    jac_colors::Vector{Int64}                     # цвет каждого столбца


    function MBSystem2D()
        bodies = Vector{AbstractBody2D}([])
        connectors = Vector{AbstractJoint2D}([])
        forces = Vector{AbstractForce2D}([])
        sensors = Vector{AbstractSensor2D}([])
        bodiesdofs = Vector{Int64}([])
        lmdofs = Vector{Int64}([])
        assembeld = false
        default_rhs = (x) -> nothing
        default_jacobian = (x) -> nothing
        default_prestep = (x) -> nothing
        default_measure = (measurements, state) -> nothing
        default_kinematic_constrains = (residual, q) -> nothing
        targets = Vector{Float64}([])
        mass = Matrix{Float64}([;;])
        return new(
            bodies,
            connectors, 
            forces,
            sensors,
            bodiesdofs, 
            lmdofs,
            assembeld,
            default_rhs, 
            default_jacobian,
            default_prestep, 
            default_measure,
            default_kinematic_constrains, 
            targets,
            mass,
            spzeros(0, 0),
            Int64[])
    end
end

get_mass_matrix(sys::MBSystem2D) = sys.mass
number_of_bodies(sys::MBSystem2D) = length(sys.bodies)
bodies(sys::MBSystem2D) = sys.bodies
joints(sys::MBSystem2D) = sys.joints
connectors(sys::MBSystem2D) = sys.connectors

get_targets(sys::MBSystem2D) = sys.targets

function update_targets!(sys::MBSystem2D)
    
    for connector in sys.connectors
        propagate_targets!(sys, connector)
    end
end

function last_body_dof(sys::MBSystem2D)
    if (isempty(sys.bodiesdofs))
        return 0
    end
    return (sys.bodiesdofs)[end]
end
function last_lm_dof(sys::MBSystem2D)
    if (isempty(sys.lmdofs))
        return last_body_dof(sys)
    else
        return sys.lmdofs[end]
    end
end

number_of_dofs(sys) = last_lm_dof(sys) + 1
number_of_generalized_coordinates(sys) = length(sys.bodies) * 3


function assemble!(sys; sparse_jacobian::Bool=true)
    state_length = number_of_dofs(sys) 
    sys.mass = zeros(state_length, state_length)

    for body in sys.bodies

        last_dof = sys.bodiesdofs[body.index]

        position_dofs = [
            last_dof - 5,
            last_dof - 4,
            last_dof - 3,
        ]

        velocity_dofs = [
            last_dof - 2,
            last_dof - 1,
            last_dof,
        ]

        sys.mass[position_dofs, position_dofs] = I(3)
        sys.mass[velocity_dofs, velocity_dofs] = Diagonal([body.mass, body.mass, body.inertia])
    end  
    sys.mass[end, end] = 1.0
    sys.targets = zeros(number_of_dofs(sys))
    update_targets!(sys)
    
    sys.rhs = (state) -> begin
        ret = similar(state)
        fill!(ret, zero(state[1]))
        for body in sys.bodies
            add_to_rhs!(ret, state, sys, body)
        end
        for connector in connectors(sys)
            add_to_rhs!(ret, state, sys, connector)
        end
        for force in sys.forces
            add_to_rhs!(ret, state, sys, force)
        end
        ret -= get_targets(sys)
        ret[end] = 1.0
        return ret
    end

    if sparse_jacobian
        # шаблон и раскраску считаем один раз, они не зависят от состояния
        sys.jac_pattern = find_jacobian_pattern(sys)
        sys.jac_colors = color_columns(sys.jac_pattern)
        sys.jacobian = (state) -> calc_sparse_jacobian(sys, state)
    else
        # старый вариант, оставила чтобы сравнивать
        sys.jacobian = (state) -> ForwardDiff.jacobian(sys.rhs, state)
    end
    sys.kinematic_constrains! = (residual, generalized_coordinates) -> begin
        for connector in connectors(sys)
            compute_kinematic_residual!(residual, generalized_coordinates, sys, connector)
        end
    end
    lbd = last_body_dof(sys)
    de_ = 1:lbd;
    mass_de_factorization = lu(sys.mass[de_, de_])
    sys.measure! = (measurements::Vector{Float64}, state::Vector{Float64}) -> begin
        current_rhs = sys.rhs(state)
        dstate = similar(state)
        fill!(dstate, zero(state[1]))

        dstate[de_] = mass_de_factorization \ current_rhs[de_]
        
        for s_id in eachindex(sys.sensors)
            measurements[s_id] = measure(state, dstate, sys, sys.sensors[s_id])
        end
        return nothing
    end

    sys.assembled = true
end

function set_initial_position!(
    initial::Vector{Float64}, sys::MBSystem2D, 
    body::BT, values::SVector{3, Float64}) where BT <: AbstractBody2D
    dofs = get_body_position_dofs(sys, body)
    initial[dofs] .= values
end

function set_initial_velocity!(
    initial::Vector{Float64}, sys::MBSystem2D, 
    body::BT, values::SVector{3, Float64}) where BT <: AbstractBody2D
    dofs = get_body_velocity_dofs(sys, body)
    initial[dofs] .= values
end

function init_measurement(sys::MBSystem2D, state::Matrix{Float64})
    meas_len = length(sensors);
    return Matrix{Float64}(undef, meas_len, size(state, 2);)
end




function update_act_jacobian!(sys::MBSystem2D, jac::AbstractMatrix{Float64})
    for connector in connectors(sys)
        update_jacobian!(sys, jac, connector)
    end
end