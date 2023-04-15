module SLR_Vector_Parameter

using Revise    # Allows re-running "using SLR_Vector_Parameter" after changes to this file
using Optim     # For optimization
using Distributions # For multivariate normal distribution
using LinearAlgebra # For inverse and diag functions
using ProgressMeter

include("Helper_Functions.jl")
include("Obs_Data_Likelihood_Functions.jl")
include("Complete_Data_Likelihood_Functions.jl")
include("Conditional_Distribution_Functions.jl")
include("EM_Functions.jl")
include("MCEM_Functions.jl")
include("Ascent_MCEM_Functions.jl")

# todo - Fix EM SE formula. After doing so, search for Fix-SE across all files.

end # module SLR_Vector_Parameter

