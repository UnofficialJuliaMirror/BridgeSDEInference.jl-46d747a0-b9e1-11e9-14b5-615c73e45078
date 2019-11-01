
### INFO
# I want to do inference on 4 parameters at index 4, 5, 10, 12
# The remaining 8 parameters are fixed
# As a first step, I want to do it in the following preliminary setting
# withouht blocking, with known initial observation,  and with one MH step
# (all these setting will be changed once the algorithm works)


SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)
include(joinpath(AUX_DIR, "read_and_write_data.jl"))
include(joinpath(AUX_DIR, "read_JR_data.jl"))
include(joinpath(AUX_DIR, "transforms.jl"))

# Using BridgeSDEInference
using BridgeSDEInference
using Distributions # to define priors
using Random        # to seed the random number generator
using DataFrames
using CSV


# Fetch the data
fptObsFlag = false
filename = "jr_path_part_obs.csv"
init_obs = "jr_initial_obs.csv"
(df, x0, obs, obs_time, fpt,
      fptOrPartObs) = readDataJRmodel(Val(fptObsFlag), joinpath(OUT_DIR, filename))

# Initial parameter guess.
θ₀ = [3.25, 0.1, 22.0, 0.05 , 135.0, 5.0, 6.0, 0.56, 0.0, 220.0, 0.0, 2000.0]

# P_Target
Pˣ = JRNeuralDiffusion(θ₀...)
#P_auxiliary
P̃ = [JRNeuralDiffusionAux2(θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs[1:end-1], obs[2:end])]

setup = MCMCSetup(Pˣ,P̃, PartObs())

# Observation operator and noise
L = @SMatrix [0. 1. -1. 0. 0. 0.]
Σdiagel = 10^(-10)
Σ = @SMatrix [Σdiagel]
set_observations!(setup, [L for _ in obs], [Σ for _ in obs], obs, obs_time)

# Obsevration frequency
obs_time[2] - obs_time[1]
# Imputation grid < observation frequency
dt = 0.0001
set_imputation_grid!(setup, dt)

# Parameter update
param_updt = true

# Memory paramter of the preconditioned Crank Nicolson scheme
pCN = 0.9

# Inference for  b (positive), C (positive), μ_y, σ_y (positive)
positive = [1,2,4]

# Transition kernel
# ?should the RandomWalk be defined for all the 12th parameters?
 t_kernel = [RandomWalk(fill(0.1, 4), positive)]

# Update parameter list
BridgeSDEInference.param_names(Pˣ)
updt_coord = ((4, 5, 10, 12))

# Update type
updt_type = (MetropolisHastingsUpdt(), )

set_transition_kernels!(setup, t_kernel,
                        pCN,
                        param_updt,
                        updt_coord,
                        updt_type)

# Starting Point
x0 = ℝ{6}(0.08, 18, 15, -0.5, 0, 0)
x0Pr = KnownStartingPt(x0)

# Automatic assignment of indecesForUpdt
n = 4 # number of parameters being update
priors_par = Priors([ImproperPrior() for i in 1:n])
set_priors!(setup, priors_par, x0Pr)


# No Blocking (defualt)
set_blocking!(setup)

# ODE solvers
set_solver!(setup)
#  MCMC parameters
num_mcmc_steps = 10000
save_iter = 3*10^2
verb_iter = 10^2
skip_for_save = 1
warm_up = 100
set_mcmc_params!(setup, num_mcmc_steps, save_iter,
                 verb_iter, skip_for_save, warm_up)

# Initialisation of internal containers
initialise!(eltype(x0), setup, true)


out = mcmc(setup)