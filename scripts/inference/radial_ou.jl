# ------------------------------------------------------------
# NOTE BROKEN AND POSSIBLY BEYOND REPAIR                     |
# explosion of likelihoods seems to make inference impossible|
# ------------------------------------------------------------

SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))


using StaticArrays
using Distributions # to define priors
using Random        # to seed the random number generator

# Let's generate the data
# -----------------------
using Bridge
#import Main.BridgeSDEInference.forcedSolve
include(joinpath(SRC_DIR, "auxiliary", "data_simulation_fns.jl"))
Random.seed!(4)
θ₀ = [0.05, √2.0]
Pˣ = RadialOU(θ₀...)

x0, dt, T = ℝ{1}(0.5), 1/5000, 1.0
tt = 0.0:dt:T
XX, _ = simulateSegment(0.0, x0, Pˣ, tt)

num_obs = 11
skip = div(length(tt), num_obs-1)
obs_time, obs_vals = collect(tt)[1:skip:end], XX.yy[1:skip:end]
P̃ = [RadialOUAux(θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obsTime[1:end-1], obsTime[2:end], obsVals[1:end-1], obsVals[2:end])]
L = @SMatrix [1.]
Σ = @SMatrix [10^(-3)]

setup = MCMCSetup(Pˣ, P̃, PartObs())
set_observations!(setup, [L for _ in P̃], [Σ for _ in P̃])
set_imputation_grid!(setup, 1/1000)
set_transition_kernels!(setup,
                        [RandomWalk([0.002, 0.1], [true, true]),
                         RandomWalk([0.002, 0.1], [true, true])],
                        0.975, true,
                        (Val((true, false)),
                         Val((false, true)),
                        ),
                        (MetropolisHastingsUpdt(),
                         MetropolisHastingsUpdt(),
                        ))
set_priors!(setup, Priors((ImproperPrior(), ImproperPrior())), KnownStartingPt(x0))
set_mcmc_params!(setup, 1*10^1, 1*10^0, 10^2, 10^0, 0)
set_blocking!(setup)
set_solver!(setup, Vern7(), NoChangePt())
initialise!(eltype(x0), setup)

Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)

Xs = [[x[1] for x in path] for path in paths]

using Plots
p = plot(time_, Xs[1], color="steelblue", alpha=0.5, label="", ylims=[0,5])
scatter!(obsTime, [x[1] for x in obsVals])
display(p)
