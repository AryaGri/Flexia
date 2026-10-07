module Flexia
# using Pkg; Pkg.activate(".")
using StaticArrays
using Makie
using LinearAlgebra
using ForwardDiff
using SparseArrays
using SparseConnectivityTracer: TracerSparsityDetector, jacobian_sparsity
# export MBSystem
# export Body2D



export AbstractBody2D, AbstractConnector2D, AbstractActuator2D, AbstractJoint2D, AbstractPositionActuator2D, AbstractSensor2D

abstract type AbstractBody2D end
abstract type AbstractConnector2D end
abstract type AbstractForce2D end
abstract type AbstractJoint2D <: AbstractConnector2D end
abstract type AbstractActuator2D <: AbstractConnector2D end
abstract type AbstractPositionActuator2D <: AbstractActuator2D end
abstract type AbstractSensor2D end
abstract type AbstractPositionSensor <: AbstractSensor2D end
abstract type AbstractVelocitySensor <: AbstractSensor2D end
abstract type AbstractAccelerationSensor <: AbstractSensor2D end


abstract type AbstractAxis end
struct XAxis <: AbstractAxis end
struct YAxis <: AbstractAxis end

export Body2D
export FixedJoint, HingeJoint, SliderJoint, TorsionalSpring, LinearSpring
export PositionMotor2D, PositionLinearActuator2D
export MBSystem2D

# connectors
export set_position_on_first_body!, set_position_on_second_body!, set_direction_on_first_body!, set_direction_on_second_body!, setposition!, setrotation!
# bodies
export bodies, number_of_bodies, connectors

export get_body_position_dofs, get_body_velocity_dofs, number_of_dofs, last_body_dof, last_lm_dof, get_boundary_points, get_lms, get_spring_moment, get_spring_energy, number_of_generalized_coordinates, get_body_generalized_dofs

# forces
export BodyTimeVariableForce, set_force!, set_position!

# solve
export cros!, static_solver!, simulate, find_natural_freqs

# Actuators
export settarget!

# Sensors
export FramePositionSensor, FrameGlobalVelocitySensor, FrameLocalVelocitySensor, FrameLocalAccelerationSensor

# system
export add!, assemble!, get_mass_matrix
export set_initial_position!, set_initial_velocity!

# visualization
export animate, draw_static

# utils
export set_initial_state_value!, set_pos_state_value! 

export test_func


test_func() = 1

include("sparse_jacobian.jl")
include("system.jl")
include("solvers.jl")
# include("markers.jl")
include("bodies.jl")
include("connectors.jl")
include("joints.jl")
include("forces.jl")
include("actuators.jl")
include("sensors.jl")
include("visualize.jl")
include("utils.jl")

function getdofs(sys::MBSystem2D, body::Body2D)
    if (body.index == -1)
        return 0:0
    elseif (body.index == 1)
        return 1:sys.bodiesdofs[1]
    else
        return (sys.bodiesdofs[body.index-1]+1):sys.bodiesdofs[body.index]
    end
end


end



"""
# end # module Flexia

"""