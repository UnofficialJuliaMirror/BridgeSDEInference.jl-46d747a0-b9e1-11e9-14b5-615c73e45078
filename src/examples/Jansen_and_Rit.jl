#Stage 1(a)
using Bridge
using StaticArrays
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N, T} where {N, T}

"""
    JRNeuralDiffusion <: ContinuousTimeProcess{ℝ{6, T}}
structure defining the Jansen and Rit Neural Mass Model described in
https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0046-4 and
https://arxiv.org/abs/1903.01138
"""
struct JRNeuralDiffusion{T} <: ContinuousTimeProcess{ℝ{6, T}}
    A::T
    a::T
    B::T
    b::T
    C::T
    νmax::T
    v0::T
    r::T
    μx::T
    μy::T
    μz::T
    σy::T
    # constructor given assumption statistical paper
    function JRNeuralDiffusion(A::T, a::T, B::T, b::T, C::T,
            νmax::T, v0::T ,r::T, μx::T, μy::T, μz::T, σy::T) where T
        new{T}(A, a, B, b, C, νmax, v0, r, μx, μy, μz, σy)
    end
end
#C1 = C, C2 = 0.8C, c3 = 0.25C, c4 =  0.25C,

# in the statistical paper they set μ's to be constant and not function of time.
function μx(t, P::JRNeuralDiffusion{T}) where T
    P.μx
end

function μy(t, P::JRNeuralDiffusion{T}) where T
    P.μy
end

function μz(t, P::JRNeuralDiffusion{T}) where T
    P.μz
end

"""
    sigm(x, P::JRNeuralDiffusion)
definition of sigmoid function
"""
function sigm(x, P::JRNeuralDiffusion{T}) where T
    P.νmax / (1 + exp(P.r*(P.v0 - x)))
end


function b(t, x, P::JRNeuralDiffusion{T}) where T
    ℝ{6}(x[4], x[5], x[6],
    P.A*P.a*(μx(t, P) + sigm(x[2] - x[3], P)) - 2P.a*x[4] - P.a*P.a*x[1],
    P.A*P.a*(μy(t, P) + 0.8P.C*sigm(P.C*x[1], P)) - 2P.a*x[5] - P.a*P.a*x[2],
    P.B*P.b*(μz(t, P) + 0.25P.C*sigm(0.25P.C*x[1], P)) - 2P.b*x[6] - P.b*P.b*x[3])
end


#6x1 matrix
function σ(t, x, P::JRNeuralDiffusion{T}) where T
    @SMatrix    [0.0;
                0.0 ;
                0.0 ;
                0.0 ;
                P.σy ;
                0.0 ]
end

constdiff(::JRNeuralDiffusion) = true
clone(::JRNeuralDiffusion, θ) = JRNeuralDiffusion(θ...)
params(P::JRNeuralDiffusion) = [P.A, P.a, P.B, P.b, P.C, P.νmax,
    P.v0, P.r, P.μx, P.μy, P.μz, P.σy]
param_names(::JRNeuralDiffusion) = (:A, :a, :B, :b, :C, :νmax,
    :v0, :r, :μx, :μy, :μz, :σy)
#auxiliary process
"""
    JRNeuralDiffusionaAux1{T, S1, S2} <: ContinuousTimeProcess{ℝ{6, T}}
structure for the auxiliary process (defined as linearized process at the final point)
"""
struct JRNeuralDiffusionAux1{R, S1, S2} <: ContinuousTimeProcess{ℝ{6, R}}
    A::R
    a::R
    B::R
    b::R
    C::R
    νmax::R
    v0::R
    r::R
    μx::R
    μy::R
    μz::R
    σy::R
    u::S1
    t::Float64
    v::S2
    T::Float64
    # constructor given assumptions paper
    function JRNeuralDiffusionAux1(A::R, a::R, B::R, b::R, C::R,
                        νmax::R, v0::R ,r::R, σy::R, t::Float64, u::S1,
                        T::Float64, v::S2) where {R, S1, S2}
        new{R, S1, S2}(A, a, B, b, C, νmax, v0, r, σy, t, u, T, v)
    end
end



"""
    sigm(x, P::JRNeuralDiffusionAux1)
definition of sigmoid function
"""
function sigm(x, P::JRNeuralDiffusionAux1{T}) where T
    P.νmax / (1 + exp(P.r*(P.v0 - x)))
end

"""
    d1sigm(x, P::JRNeuralDiffusionAux1{T, S1, S2})
derivative of sigmoid function
"""
function d1sigm(x, P::JRNeuralDiffusionAux1{T, S1, S2}) where {T, S1, S2}
    P.νmax*P.r*exp(P.r*(P.v0 - x))/(1 + exp(P.r*(P.v0 - x)))^2
end

function μx(t, P::JRNeuralDiffusionAux1{T}) where T
    P.μx
end

function μy(t, P::JRNeuralDiffusionAux1{T}) where T
    P.μy
end

function μz(t, P::JRNeuralDiffusionAux1{T}) where T
    P.μz
end




function B(t, P::JRNeuralDiffusionAux1{T, S1, S2}) where {T, S1, S2}
    @SMatrix [0.0  0.0  0.0  1.0  0.0  0.0;
              0.0  0.0  0.0  0.0  1.0  0.0;
              0.0  0.0  0.0  0.0  0.0  1.0;
              -P.a*P.a  P.A*P.a*d1sigm(P.v[2] - P.v[3], P)  -P.A*P.a*d1sigm(P.v[2] - P.v[3], P)   -2P.a  0.0  0.0;
              P.A*P.a*P.C*0.8P.C*d1sigm(P.C*P.v[1], P)  -P.a*P.a  0.0  0.0  -2P.a  0.0;
              P.B*P.b*0.25P.C*0.25P.C*d1sigm(0.25P.C*P.v[1], P)  0.0  -P.b*P.b  0.0  0.0  -2P.b]
end


function β(t, P::JRNeuralDiffusionAux1{T, S1, S2}) where {T, S1, S2}
    ℝ{6}(0.0, 0.0, 0.0,
        P.A*P.a*(μx(t, P) + sigm(P.v[2] - P.v[3], P) - d1sigm(P.v[2] - P.v[3], P)*(P.v[2] - P.v[3])),
        P.A*P.a*(μy(t, P) + 0.8P.C*(sigm(P.C*P.v[1], P) - d1sigm(P.C1*P.v[1], P)*(P.C1*P.v[1]))),
        P.B*P.b*(μz(t, P) + 0.25P.C*(sigm(0.25P.C*P.v[1], P) - d1sigm(0.25P.C*P.v[1], P)*(0.25P.C*P.v[1]))) )
end

function σ(t, P::JRNeuralDiffusionAux1{T, S1, S2}) where {T, S1, S2}
    @SMatrix     [0.0 ;
                0.0  ;
                0.0  ;
                0.0 ;
                 P.σy;
                0.0 ]
end



b(t, x, P::JRNeuralDiffusionAux1) = B(t,P) * x + β(t,P)
a(t, P::JRNeuralDiffusionAux1) = σ(t,P) * σ(t, P)'

constdiff(::JRNeuralDiffusionAux1) = true
clone(P::JRNeuralDiffusionAux1, θ) = JRNeuralDiffusionAux1(θ..., P.t, P.u, P.T, P.v)
clone(P::JRNeuralDiffusionAux1, θ, v) = JRNeuralDiffusionAux1(θ..., P.t, zero(v), P.T, v)
depends_on_params(::JRNeuralDiffusionAux1) = (1,2,3,4,5,6,7,8,9,10,11,12)


params(P::JRNeuralDiffusionAux1) = (P.A, P.a, P.B, P.b, P.C, P.νmax,
                                        P.v0, P.r, P.μx, P.μy, P.μz, P.σy)

param_names(P::JRNeuralDiffusionAux1) = (:A, :a, :B, :b, :C, :νmax,
                                                        :v0, :r, :μx, :μy, :μz, :σy)

"""
    JRNeuralDiffusionaAux2{T, S1, S2} <: ContinuousTimeProcess{ℝ{6, T}}
structure for the auxiliary process defined as linearized process at the final point
for the random variable V_t = LX_t and around the point tt in ℝ¹ (user choice, if not
specified around v0) for the unobserved first components.
"""
struct JRNeuralDiffusionAux2{R, S1, S2} <: ContinuousTimeProcess{ℝ{6, R}}
    tt::R
    A::R
    a::R
    B::R
    b::R
    C::R
    νmax::R
    v0::R
    r::R
    μx::R
    μy::R
    μz::R
    σy::R
    u::S1
    t::Float64
    v::S2
    T::Float64
    # generator given assumptions paper
    function JRNeuralDiffusionAux2(A::R, a::R, B::R, b::R, C::R,
            νmax::R, v0::R ,r::R, μx::R, μy::R, μz::R,  σy::R, t::Float64, u::S1,
                        T::Float64, v::S2; tt = v0) where {R, S1, S2}
        new{R, S1, S2}(tt, A, a, B, b, C, νmax, v0, r, μx, μy, μz, σy, t, u, T, v)
    end
end


"""
    sigm(x, P::JRNeuralDiffusionAux2)
definition of sigmoid function
"""
function sigm(x, P::JRNeuralDiffusionAux2{T}) where T
    P.νmax / (1 + exp(P.r*(P.v0 - x)))
end

"""
    d1sigm(x, P::JRNeuralDiffusionAux2{T, S1, S2})
derivative of sigmoid function
"""
function d1sigm(x, P::JRNeuralDiffusionAux2{T, S1, S2}) where {T, S1, S2}
    P.νmax*P.r*exp(P.r*(P.v0 - x))/(1 + exp(P.r*(P.v0 - x)))^2
end

function μx(t, P::JRNeuralDiffusionAux2{T}) where T
    P.μx
end

function μy(t, P::JRNeuralDiffusionAux2{T}) where T
    P.μy
end

function μz(t, P::JRNeuralDiffusionAux2{T}) where T
    P.μz
end


function B(t, P::JRNeuralDiffusionAux2{T, S1, S2}) where {T, S1, S2}
    @SMatrix [0.0  0.0  0.0  1.0  0.0  0.0;
              0.0  0.0  0.0  0.0  1.0  0.0;
              0.0  0.0  0.0  0.0  0.0  1.0;
              -P.a*P.a  P.A*P.a*d1sigm(P.v[1], P)  -P.A*P.a*d1sigm(P.v[1], P)   -2P.a  0.0  0.0;
              P.A*P.a*P.C*0.8P.C*d1sigm(P.C*P.tt, P)  -P.a*P.a  0.0  0.0  -2P.a  0.0;
              P.B*P.b*0.25P.C*0.25P.C*d1sigm(0.25P.C*P.tt, P)  0.0  -P.b*P.b  0.0  0.0  -2P.b]
end


function β(t, P::JRNeuralDiffusionAux2{T, S1, S2}) where {T, S1, S2}
    ℝ{6}(0.0, 0.0, 0.0,
        P.A*P.a*(μx(t, P) + sigm(P.v[1], P) - d1sigm(P.v[1], P)*(P.v[1])),
        P.A*P.a*(μy(t, P) + 0.8P.C*(sigm(P.C*P.tt, P) - d1sigm(P.C*P.tt, P)*(P.C*P.tt))),
        P.B*P.b*(μz(t, P) + 0.25P.C*(sigm(0.25P.C*P.tt, P) - d1sigm(0.25P.C*P.tt, P)*(0.25P.C*P.tt))) )
end

function σ(t, P::JRNeuralDiffusionAux2{T, S1, S2}) where {T, S1, S2}
    @SMatrix    [0.0 ;
                0.0  ;
                0.0  ;
                0.0 ;
                 P.σy;
                0.0 ]
end


b(t, x, P::JRNeuralDiffusionAux2) = B(t,P) * x + β(t,P)
a(t, P::JRNeuralDiffusionAux2) = σ(t,P) * σ(t, P)'

constdiff(::JRNeuralDiffusionAux2) = true
clone(P::JRNeuralDiffusionAux2, θ) = JRNeuralDiffusionAux2(θ..., P.t, P.u, P.T, P.v)
clone(P::JRNeuralDiffusionAux2, θ, v) = JRNeuralDiffusionAux2(θ..., P.t, zero(v), P.T, v)
params(P::JRNeuralDiffusionAux2) = (P.A, P.a, P.B, P.b, P.C, P.νmax,
                                        P.v0, P.r, P.μx, P.μy, P.μz, P.σy)
param_names(P::JRNeuralDiffusionAux2) = (:A, :a, :B, :b, :C, :νmax,
                                        :v0, :r, :μx, :μy, :μz, :σy)

depends_on_params(::JRNeuralDiffusionAux2) = (1,2,3,4,5,6,7,8,9,10,11,12)
