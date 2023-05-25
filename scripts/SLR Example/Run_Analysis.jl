
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
obs_info = obs_data_obs_info(theta_MLE,Y)
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

theta_hat_MCEM1, theta_hat_traj_MCEM1 = run_MCEM_fixed_iteration_count(theta_init, Y, M, K; return_trajectory=true)
cov_MCEM1 = MCEM_cov_formula_iid(theta_hat_MCEM1, Y, M, false)

# ----------------------------- Plot Trajectories ---------------------------- #
p_hat_traj1 = getindex.(theta_hat_traj_MCEM1, 1)
q_hat_traj1 = getindex.(theta_hat_traj_MCEM1, 2)

naive_MCEM_plot1 = plot(p_hat_traj1, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:right, legendfont=font(20), guidefont=font(20))
plot!(naive_MCEM_plot1, q_hat_traj1, label = "q")
# hline!(naive_MCEM_plot1, [theta_MLE], label = nothing)
savefig(naive_MCEM_plot1, plotsdir("Blood_Type", "naive_MCEM_traj1.pdf"))


# --------------------------- Re-Run With Larger M --------------------------- #
M_large = 1000
K_new = 20

theta_hat_MCEM2, theta_hat_traj_MCEM2 = run_MCEM_fixed_iteration_count(theta_hat_MCEM1, Y, M_large, K_new; return_trajectory=true)
cov_MCEM2 = MCEM_cov_formula_iid(theta_hat_MCEM2, Y, M, false)

p_hat_traj2 = getindex.(theta_hat_traj_MCEM2, 1)
q_hat_traj2 = getindex.(theta_hat_traj_MCEM2, 2)

naive_MCEM_plot2 = plot(p_hat_traj2, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:right, legendfont=font(20), guidefont=font(20))
plot!(naive_MCEM_plot2, q_hat_traj2, label = "q")
savefig(naive_MCEM_plot2, plotsdir("Blood_Type", "naive_MCEM_traj2.pdf"))


# ---------------------------------------------------------------------------- #
#                                Analysis: AMCEM                               #
# ---------------------------------------------------------------------------- #

alpha1 = 0.2
alpha2 = 0.2
alpha3 = 0.1
k = 2
atol = 1e-1
AMCEM_control = Ascent_MCEM_Control(alpha1, alpha2, alpha3, k, atol)
ascent_MCEM_control = AMCEM_control

M_init = 10


some_theta_AMCEMs = []

theta_AMCEM, all_theta_hat_AMCEMs = run_ascent_MCEM([0.1, 0.1], Y, M_init, AMCEM_control; diagnostics=true)



p_hat_traj_AMCEM = getindex.(all_theta_hat_AMCEMs, 1)
q_hat_traj_AMCEM = getindex.(all_theta_hat_AMCEMs, 2)

AMCEM_plot = plot(p_hat_traj_AMCEM, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:right, legendfont=font(20), guidefont=font(20));
plot!(AMCEM_plot, q_hat_traj_AMCEM, label = "q");
hline!(AMCEM_plot, [theta_MLE], label = nothing)
