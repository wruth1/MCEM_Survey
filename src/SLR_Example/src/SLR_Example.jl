
module SLR_Example


using Revise    # Allows re-running "using SLR_Example" after changes to this file
using Distributions # For multivariate normal distribution
using LinearAlgebra # For inverse and diag functions
using ProgressMeter
using LogExpFunctions   # For logsumexp function
using JuMP, Ipopt   # For optimization. Optim only allows box constraints.

include("Helper_Functions.jl")
include("Obs_Data_Likelihood_Functions.jl")
include("Complete_Data_Likelihood_Functions.jl")
include("Conditional_Distribution_Functions.jl")
include("EM_Functions.jl")
include("MCEM_Functions.jl")
include("Ascent_MCEM_Functions.jl")
include("Booth_Hobert.jl")
include("Chan_Ledolter.jl")




end # module SLR_Example
