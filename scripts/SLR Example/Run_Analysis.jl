
using Pkg
using DrWatson

push!(LOAD_PATH, srcdir("SLR_Example", "src"));

using SLR_Example


using Random
using Distributions
using LinearAlgebra
using ProgressMeter

using Plots
using Measures  # For plot margins

# Define some functions to help navigate the project directory
SLRdir(args...) = srcdir("SLR_Example", args...)
SLRsrcdir(args...) = SLRdir("src", args...)
SLRtestdir(args...) = SLRdir("test", args...)

# Run this before testing to make sure that Julia looks in the right place for tests
# Pkg.activate(srcdir("SLR_Example"))

# Run this to edit the packages used for testing
# Pkg.activate(srcdir("SLR_Example", "test"))


# ---------------------------------------------------------------------------- #
#                              Construct Some Data                             #
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
theta2 = [0.2, 0.4]


# ---------------------------------------------------------------------------- #
#                             Actual Data Analysis                             #
# ---------------------------------------------------------------------------- #

# # Full dataset
# # Sample size is so large that variability is negligible
# Y = [1305924, 1725950, 988996, 444479]
# sum(Y) - 4465349

# # A smaller sample from the city of Yame in Fukuoka Prefecture
# # Yame is where Gyokuru is grown
# Y = [664, 1092, 538, 257]

# # A very small sample (comparatively), from the city of Tonosho in Kagawa Prefecture
# Y = [49, 80, 28, 11]

# # A very very small sample, from the administrative division of Oto in Nara Prefecture
Y = [10, 16, 7, 1]

theta_init = [1/3, 1/3]


# ---------------------------------------------------------------------------- #
#                                 Analysis: MLE                                #
# ---------------------------------------------------------------------------- #

theta_MLE = obs_data_MLE(Y)
cov_MLE = obs_data_MLE_Cov(theta_MLE, Y)



# ---------------------------------------------------------------------------- #
#                                 Analysis: EM                                 #
# ---------------------------------------------------------------------------- #

# ----------------------------- Compute estimate ----------------------------- #
theta_hat_EM, theta_hat_traj_EM = run_EM(theta_init, Y, rtol = 1e-8, return_trajectory=true)
p_hat_EM, q_hat_EM = theta_hat_EM
r_hat_EM = 1 - p_hat_EM - q_hat_EM

# ----------------------------- Plot Trajectories ---------------------------- #
p_hat_traj = getindex.(theta_hat_traj_EM, 1)
q_hat_traj = getindex.(theta_hat_traj_EM, 2)

EM_plot = plot(p_hat_traj, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:right, legendfont=font(20), guidefont=font(20))
plot!(EM_plot, q_hat_traj, label = "q")
scatter!(EM_plot, p_hat_traj, label = nothing)
scatter!(EM_plot, q_hat_traj, label = nothing)
savefig(EM_plot, plotsdir("Blood_Type", "EM_traj.pdf"))

# -------------------------------- Estimate SE ------------------------------- #
obs_info_EM = EM_obs_data_information_formula(theta_hat_EM, Y)
cov_EM = EM_COV_formula(theta_hat_EM, Y)

cov_EM - cov_MLE



# ---------------------------------------------------------------------------- #
#                             Analysis: Naive MCEM                             #
# ---------------------------------------------------------------------------- #

# Fixed MC size for each iteration
M = 100

# Preliminary number of iterations
K = 50

theta_hat_MCEM, theta_hat_traj_MCEM = run_MCEM_fixed_iteration_count(theta_init, Y, M, K; return_trajectory=true)
cov_MCEM = MCEM_cov_formula_iid(theta_hat_MCEM, Y, M, false)

# ----------------------------- Plot Trajectories ---------------------------- #
p_hat_traj = getindex.(theta_hat_traj_MCEM, 1)
q_hat_traj = getindex.(theta_hat_traj_MCEM, 2)

naive_MCEM_plot = plot(p_hat_traj, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:right, legendfont=font(20), guidefont=font(20))
plot!(naive_MCEM_plot, q_hat_traj, label = "q")
savefig(naive_MCEM_plot, plotsdir("Blood_Type", "naive_MCEM_traj.pdf"))



# ---------------------------------------------------------------------------- #
#                                Analysis: AMCEM                               #
# ---------------------------------------------------------------------------- #

alpha1 = 0.2
alpha2 = 0.2
alpha3 = 0.3
k = 2
atol = 1e-2
AMCEM_control = Ascent_MCEM_Control(alpha1, alpha2, alpha3, k, atol)

M_init = 500


some_theta_AMCEMs = []

theta_AMCEM = run_ascent_MCEM([0.1, 0.1], Y, theta_fixed, M_init, AMCEM_control)
push!(some_theta_AMCEMs, theta_AMCEM)
