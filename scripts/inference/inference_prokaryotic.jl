SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
#include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))
#include(joinpath(SRC_DIR, "fitzHughNagumo_conjugateUpdt.jl"))

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "vern7.jl"))
include(joinpath(SRC_DIR, "tsit5.jl"))
include(joinpath(SRC_DIR, "rk4.jl"))
include(joinpath(SRC_DIR, "ralston3.jl"))
include(joinpath(SRC_DIR, "priors.jl"))
include(joinpath(SRC_DIR, "guid_prop_bridge.jl"))

include(joinpath(SRC_DIR, "bounded_diffusion_domain.jl"))
include(joinpath(SRC_DIR, "radial_ornstein_uhlenbeck.jl"))
include(joinpath(SRC_DIR, "euler_maruyama_dom_restr.jl"))
include(joinpath(SRC_DIR, "prokaryotic_autoregulatory_gene_network.jl"))

include(joinpath(SRC_DIR, "random_walk.jl"))
include(joinpath(SRC_DIR, "blocking_schedule.jl"))
include(joinpath(SRC_DIR, "starting_pt.jl"))
include(joinpath(SRC_DIR, "mcmc.jl"))
include(joinpath(SRC_DIR, "path_to_wiener.jl"))


using StaticArrays
using Distributions # to define priors
using Random        # to seed the random number generator

# Let's generate the data
# -----------------------
using Bridge
include(joinpath(AUX_DIR, "data_simulation_fns.jl"))
Random.seed!(4)
# values taken from table 1 of Golithly and Wilkinson
θ₀ = [0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1]
K = 10.0
Pˣ = Prokaryote(θ₀..., K)

x0, dt, T = ℝ{4}(7.0, 10.0, 4.0, 6.0), 1/5000, 20.0
tt = 0.0:dt:T
XX, _ = simulateSegment(ℝ{4}(0.0, 0.0, 0.0, 0.0), x0, Pˣ, tt)
skip = 2500
obsTime, obsVals = XX.tt[1:skip:end], XX.yy[1:skip:end]

fptOrPartObs = PartObs()
fpt = [NaN for _ in obsTime[2:end]]

auxFlag = Val{(true, true, true, true)}()
P̃ = [ProkaryoteAux(θ₀..., K, t₀, u, T, v, auxFlag) for (t₀, T, u, v)
     in zip(obsTime[1:end-1], obsTime[2:end], obsVals[1:end-1], obsVals[2:end])]

Σdiagel = 10^-4
Σ = SMatrix{4,4}(1.0I)*Σdiagel
L = SMatrix{4,4}(1.0I)

Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=1*10^3
saveIter=1*10^0

tKernel = RandomWalk([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                     [true, true, true, true, true, true, true, true])

priors = Priors((ImproperPrior(), ImproperPrior(), ImproperPrior(),
                 ImproperPrior(), ImproperPrior(), ImproperPrior(),
                 ImproperPrior(), ImproperPrior()))

𝔅 = NoBlocking()
blockingParams = ([], 0.1, NoChangePt())
changePt = NoChangePt()
x0Pr = KnownStartingPt(x0)
warmUp = 0

Random.seed!(4)
start = time()
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(eltype(x0), fptOrPartObs, obsVals, obsTime, x0Pr,
                         ℝ{4}(0.0, 0.0, 0.0, 0.0), Pˣ, P̃, Ls, Σs, numSteps,
                         tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.7,
                         dt=1/1000,
                         saveIter=saveIter,
                         verbIter=10^1,
                         updtCoord=(Val((true, false, false, false, false,
                                         false, false, false)),
                                    ),
                         paramUpdt=false,
                         updtType=(MetropolisHastingsUpdt(),
                                   ),
                         skipForSave=10^0,
                         blocking=𝔅,
                         blockingParams=blockingParams,
                         solver=Vern7(),
                         changePt=changePt,
                         warmUp=warmUp)
elapsed = time() - start
print("time elapsed: ", elapsed, "\n")


print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)

pTp = [[[x[i] for x in path] for path in paths] for i in 1:4]

using Plots

function plotPaths(j, obsIdxS, obsIdxE)
    idxS = div((obsIdxS-1)*skip,5)+1
    idxE = div((obsIdxE-1)*skip,5)+1
    p = plot()
    for i in 1:length(paths)
        plot!(time_[idxS:idxE], pTp[j][i][idxS:idxE], label="", color="steelblue", alpha=0.3)
    end
    scatter!(obsTime[obsIdxS:obsIdxE], [x[j] for x in obsVals][obsIdxS:obsIdxE],
             color="orange", label="")
    p
end
plotPaths(1, 1, 5)
plotPaths(2, 1, 5)
plotPaths(3, 1, 5)
plotPaths(4, 1, 5)

plot([θ[1] for θ in chain])