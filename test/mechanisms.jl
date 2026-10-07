# Mechanisms shared by the tests and the Jacobian benchmark.

using Flexia
using StaticArrays

const g = 9.81

# slider-crank with a time dependent force, see examples/crankshaft.jl
function build_crankshaft(; sparse_jacobian=true)
    ground = Body2D(1e6, 1e6; length=0.01)
    crank = Body2D(10.0, 1.0; length=1.0)
    slider = Body2D(5.0, 0.1; length=0.25)
    connector = Body2D(5.0, 0.5; length=2.0)

    crank.forces[2] = (x, t) -> -crank.mass * g
    slider.forces[2] = (x, t) -> -slider.mass * g
    slider.forces[1] = (x, t) -> 2050 * sin(4 * t)
    connector.forces[2] = (x, t) -> -connector.mass * g

    sys = MBSystem2D()
    add!(sys, ground)
    add!(sys, crank)
    add!(sys, connector)
    add!(sys, slider)

    ground_joint = FixedJoint(ground)
    crank_ground = HingeJoint(ground, crank)
    set_position_on_second_body!(crank_ground, SA[-0.5, 0.0])
    crank_connector = HingeJoint(crank, connector)
    set_position_on_first_body!(crank_connector, SA[0.5, 0.0])
    set_position_on_second_body!(crank_connector, SA[-1.0, 0.0])
    connector_slider = HingeJoint(connector, slider)
    set_position_on_first_body!(connector_slider, SA[1.0, 0.0])
    slider_ground = SliderJoint(ground, slider)
    set_position_on_first_body!(slider_ground, SA[3.0, 0.0])

    add!(sys, ground_joint)
    add!(sys, crank_ground)
    add!(sys, crank_connector)
    add!(sys, connector_slider)
    add!(sys, slider_ground)
    add!(sys, TorsionalSpring(crank_ground; stiffness=10.0, damping=0.5))

    assemble!(sys; sparse_jacobian)

    initial = zeros(number_of_dofs(sys))
    set_initial_position!(initial, sys, crank, SA[0.5, 0.0, 0.0])
    set_initial_position!(initial, sys, connector, SA[2.0, 0.0, 0.0])
    set_initial_position!(initial, sys, slider, SA[3.0, 0.0, 0.0])
    return sys, initial
end

# chain of `n` pendulums hanging from a fixed body, with torsional springs
function build_chain(n::Int; sparse_jacobian=true)
    sys = MBSystem2D()
    ground = Body2D(1.0, 1.0)
    add!(sys, ground)
    links = [Body2D(1.0, 0.1) for _ in 1:n]
    for link in links
        link.forces[2] = (x, t) -> -link.mass * g
        add!(sys, link)
    end
    add!(sys, FixedJoint(ground))
    previous = ground
    hinges = HingeJoint[]
    for (k, link) in enumerate(links)
        hinge = HingeJoint(previous, link)
        k > 1 && set_position_on_first_body!(hinge, SA[0.5, 0.0])
        set_position_on_second_body!(hinge, SA[-0.5, 0.0])
        add!(sys, hinge)
        push!(hinges, hinge)
        previous = link
    end
    for hinge in hinges
        add!(sys, TorsionalSpring(hinge; stiffness=5.0, damping=0.1))
    end
    assemble!(sys; sparse_jacobian)

    initial = zeros(number_of_dofs(sys))
    for (k, link) in enumerate(links)
        set_initial_position!(initial, sys, link, SA[k - 0.5, 0.0, 0.0])
    end
    return sys, initial
end

# guide with a slider driven by PositionLinearActuator2D, see
# examples/linear_actuator.jl; the actuator adds an entry to the Jacobian
# through `update_jacobian!`
function build_linear_actuator(; sparse_jacobian=true)
    angle = deg2rad(30.0)
    dir = SA[cos(angle), sin(angle)]
    s_target(t) = 1.0 + 0.5 * sin(2.0 * t)
    v_target(t) = 1.0 * cos(2.0 * t)

    bd1 = Body2D(1.0, 0.1)
    bd2 = Body2D(1.0, 0.1)
    bd2.forces[2] = (state, t) -> -bd2.mass * g

    jnt1 = FixedJoint(bd1)
    jnt2 = SliderJoint(bd1, bd2)
    set_direction_on_first_body!(jnt2, dir)
    set_direction_on_second_body!(jnt2, dir)
    mot = PositionLinearActuator2D(jnt2, s_target(0.0), v_target(0.0))

    sys = MBSystem2D()
    add!(sys, bd1)
    add!(sys, bd2)
    add!(sys, jnt1)
    add!(sys, jnt2)
    add!(sys, mot)
    assemble!(sys; sparse_jacobian)

    initial = zeros(number_of_dofs(sys))
    set_initial_position!(initial, sys, bd2, SA[s_target(0.0) * dir[1], s_target(0.0) * dir[2], 0.0])
    set_initial_velocity!(initial, sys, bd2, SA[v_target(0.0) * dir[1], v_target(0.0) * dir[2], 0.0])
    sys.prestep = (state) -> settarget!(mot, s_target(state[end]), v_target(state[end]))
    return sys, initial
end
