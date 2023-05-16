
using SLR_Example
using Test

using ReverseDiff
using Random
using Distributions
using Statistics
using LinearAlgebra
using Optim
using JLD2

using ProgressMeter

# ---------------------------------------------------------------------------- #
#                        Construct some data for testing                       #
# ---------------------------------------------------------------------------- #



Random.seed!(1)

p_0 = 0.2
q_0 = 0.4
r_0 = 1 - p_0 - q_0
theta0 = [p_0, q_0]

prob_vec = get_cell_probs(theta0)
X_prob_vec = get_complete_cell_probs(theta0)

n = 1000
X = rand(Multinomial(n, X_prob_vec), 1)
Y = Y_from_X(X)


theta1 = [1/3, 1/3]
theta = theta1
theta_old = theta1

all_thetas = [theta0, theta1]



# include("src\\SLR_Example\\test\\Tests-Low_Level_Functions.jl")
include("Tests-Low_Level_Functions.jl")

# include("src\\SLR_Example\\test\\Tests-Obs_&_EM_&_MCEM.jl")
include("Tests-Obs_&_EM_&_MCEM.jl")

# include("src\\SLR_Example\\test\\Tests-EM_&_MCEM.jl")
include("Tests-EM_&_MCEM.jl")


# Random.seed!(1)

# beta_0 = 1.0
# mu_0 = 1.0
# tau_0 = 1.0
# sigma_0 = 1.0

# n = 100
# X = rand(Normal(mu_0, tau_0), n)
# epsilon = rand(Normal(0, sigma_0), n)
# Y = beta_0 * X + epsilon

# x = X[1]
# y = Y[1]


# theta1 = [1.0, 1.0]
# theta = theta1
# theta2 = [2.0, 1.0]
# theta3 = [1.0, 2.0]
# theta4 = [2.0, 2.0]
# all_thetas = [theta1, theta2, theta3, theta4]
# theta_fixed = [mu_0, tau_0]




# # ---------------------------------------------------------------------------- #
# #                       Run Ascent MCEM a bunch of times                       #
# # ---------------------------------------------------------------------------- #

# # Ascent MCEM is slow. 
# # Generate a single set of runs across a bunch of datasets and extract all the relevant info. Then, pass this info to tests.

# # --------------- Set control parameters for ascent-based MCEM --------------- #
# alpha1 = 0.3
# alpha2 = 0.3
# alpha3 = 0.3
# k = 3
# atol = 1e-3 # Absolute tolerance for convergence. 1e-2 takes about a minute. 1e-3 takes about 10 minutes.

# control = Ascent_MCEM_Control(alpha1, alpha2, alpha3, k, atol)

# # ------------------------ Generate data and run MCEM ------------------------ #
# # Number of MC samples to draw at first iteration
# M_start = 10

# # Number of MC samples to use for estimating SE after convergence
# M_SE = 1000

# # Number of datasets to generate
# B_MCEM = 100


# # MCEM takes a long time. Save the results once and load the file when running tests.
# # all_theta_hat_MCEMs : vector of parameter estimates from AMCEM
# #! Fix-SE
# #! After fixing the SE functions, adjust run_many_ascent_MCEMs to return the SEs as well.
# # all_theta_hat_MCEMs = run_many_ascent_MCEMs(B_MCEM, theta1, theta, theta_fixed, M_start, control)
# # @save "test/TestData/all_theta_hat_MCEMs (tol = 1e-3).jld2" all_theta_hat_MCEMs
# # @load "test/TestData/all_theta_hat_MCEMs (tol = 1e-3).jld2" # For interactive use
# @load "TestData/all_theta_hat_MCEMs (tol = 1e-3).jld2"      # For automated use


# include("Tests-Obs_&_EM_&_MCEM.jl")
# include("Tests-EM_&_MCEM.jl")
# include("Tests-MCEM.jl")


# ### The first test suite takes much longer to run than the others. Strangely, it speeds up considerably the second time I run it. I tried profiling to see what happens and it didn't really clarify. I think this has something to do with compilation, but I definitely don't understand it. A point against the compilation theory is that this test suite still takes longer if I run it after the other two.
# # @profview include("./test/Tests-Obs_&_EM_&_MCEM.jl")
