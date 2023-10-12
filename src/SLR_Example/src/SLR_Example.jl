
module SLR_Example


using Revise    # Allows re-running "using SLR_Example" after changes to this file
using Distributions # For multivariate normal distribution
using LinearAlgebra # For inverse and diag functions
using ProgressMeter
using LogExpFunctions   # For logsumexp function
using JuMP, Ipopt   # For optimization. Optim only allows box constraints.


macro counted(f)
    name = f.args[1].args[1]
    name_str = String(name)
    body = f.args[2]
    counter_code = quote
        if !haskey(COUNTERS, $name_str)
            COUNTERS[$name_str] = 0
        end
        COUNTERS[$name_str] += 1
    end
    insert!(body.args, 1, counter_code)
    return f
end

# Counter for number of Monte Carlo samples drawn. Must be re-set to zero between methods
if ! @isdefined COUNTERS
    const COUNTERS = Dict{String, Int}()
end


include("Helper_Functions.jl")
include("Obs_Data_Likelihood_Functions.jl")
include("Complete_Data_Likelihood_Functions.jl")
include("Conditional_Distribution_Functions.jl")
include("EM_Functions.jl")
include("MCEM_Functions.jl")
include("Ascent_MCEM_Functions.jl")
include("Booth_Hobert.jl")
include("Chan_Ledolter.jl")
include("SAEM_Functions.jl")
include("MCML_Functions.jl")

export COUNTERS


end # module SLR_Example
