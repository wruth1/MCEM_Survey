
using Pkg
using DrWatson

push!(LOAD_PATH, srcdir("SLR_Example", "src"));

using SLR_Example


using Random
using Distributions
using LinearAlgebra
using ProgressMeter
using LogExpFunctions

using Plots
using Measures  # For plot margins

using BenchmarkTools




# Define some functions to help navigate the project directory
SLRdir(args...) = srcdir("SLR_Example", args...)
SLRsrcdir(args...) = SLRdir("src", args...)
SLRtestdir(args...) = SLRdir("test", args...)

# Run this before testing to make sure that Julia looks in the right place for tests
# Pkg.activate(srcdir("SLR_Example"))

# Run this to edit the packages used for testing
# Pkg.activate(srcdir("SLR_Example", "test"))


num_reps_time = 100 # Number of times to repeat each method to assess its runtime


# ---------------------------------------------------------------------------- #
#                              Construct Some Data                             #
# ---------------------------------------------------------------------------- #

# Random.seed!(1)

# p_0 = 0.2
# q_0 = 0.4
# r_0 = 1 - p_0 - q_0
# theta0 = [p_0, q_0]

# prob_vec = get_cell_probs(theta0)
# X_prob_vec = get_complete_cell_probs(theta0)

# n = 1000
# X = rand(Multinomial(n, X_prob_vec), 1)
# Y = Y_from_X(X)

# theta1 = [1/3, 1/3]
# theta = theta1
# theta2 = [0.2, 0.4]


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
cor_MLE = cov2cor(cov_MLE)



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

EM_plot = plot(p_hat_traj, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:right, legendfont=font(15), guidefont=font(20), xtickfont=font(15), ytickfont=font(15))#, xticks=[1, 5, 10, 15], xlims = (0,15))#length(p_hat_traj)])
plot!(EM_plot, q_hat_traj, label = "q")
# scatter!(EM_plot, p_hat_traj, label = nothing)
# scatter!(EM_plot, q_hat_traj, label = nothing)
hline!(EM_plot, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
savefig(EM_plot, plotsdir("Blood_Type", "EM_traj.pdf"))

# -------------------------------- Estimate SE ------------------------------- #
obs_info_EM = EM_obs_data_information_formula(theta_hat_EM, Y)
cov_EM = EM_COV_formula(theta_hat_EM, Y)

cov_EM - cov_MLE


# ---------------------------------------------------------------------------- #
#                             Analysis: Naive MCEM                             #
# ---------------------------------------------------------------------------- #

Random.seed!(1)

COUNTERS["one_X_given_Y_iid"] = 0


# Fixed MC size for each iteration
M = 100

# Preliminary number of iterations
K = 50

theta_hat_MCEM1, theta_hat_traj_MCEM1 = run_MCEM_fixed_iteration_count(theta_init, Y, M, K; return_trajectory=true)
println("Number of MC draws for Naive MCEM:  $(COUNTERS["one_X_given_Y_iid"])")


cov_MCEM1 = MCEM_cov_formula_iid(theta_hat_MCEM1, Y, M, false)

# ----------------------------- Plot Trajectories ---------------------------- #
p_hat_traj1 = getindex.(theta_hat_traj_MCEM1, 1)
q_hat_traj1 = getindex.(theta_hat_traj_MCEM1, 2)

naive_MCEM_plot1 = plot(p_hat_traj1, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:right, legendfont=font(15), guidefont=font(20), xtickfont=font(15), ytickfont=font(15));
plot!(naive_MCEM_plot1, q_hat_traj1, label = "q");
# hline!(naive_MCEM_plot1, [theta_MLE], label = nothing)
hline!(naive_MCEM_plot1, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
savefig(naive_MCEM_plot1, plotsdir("Blood_Type", "naive_MCEM_traj1.pdf"))


# --------------------------- Re-Run With Larger M --------------------------- #
M_large = 1000
K_new = 20

theta_hat_MCEM2, theta_hat_traj_MCEM2 = run_MCEM_fixed_iteration_count(theta_hat_MCEM1, Y, M_large, K_new; return_trajectory=true)
cov_MCEM2 = MCEM_cov_formula_iid(theta_hat_MCEM2, Y, M, false)

p_hat_traj2 = getindex.(theta_hat_traj_MCEM2, 1)
q_hat_traj2 = getindex.(theta_hat_traj_MCEM2, 2)

naive_MCEM_plot2 = plot(p_hat_traj2, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:right, legendfont=font(15), guidefont=font(20), xtickfont=font(15), ytickfont=font(15));
plot!(naive_MCEM_plot2, q_hat_traj2, label = "q");
hline!(naive_MCEM_plot2, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
plt.savefig(naive_MCEM_plot2, plotsdir("Blood_Type", "naive_MCEM_traj2.pdf"), bbox_inches="tight")



# ---------------------------------------------------------------------------- #
#                            Chan and Ledolter MCEM                            #
# ---------------------------------------------------------------------------- #

COUNTERS["one_X_given_Y_iid"] = 0

Random.seed!(2)


# using SLR_Example

M_pilot = 10    # MC size for MCEM iterations in pilot study
K_max = 20      # Number of MCEM iterations to use in pilot study

R_keep = 5      # Numer of estimates after maximizer to keep from pilot study
B = 5           # Number of replicate updates to take at each kept estimate

delta = 1e-3    # Allowable SE for estimate of observed data log lik rat

theta_hat_traj_CL, lik_rat_traj_CL, SE_pilot, SE_final = run_MCEM_Chan_Ledolter(Y, theta_init, M_pilot, K_max, R_keep, B, delta; diagnostics=true)
println("Number of MC draws for Chan and Ledolter MCEM:  $(COUNTERS["one_X_given_Y_iid"])")



# ----------------------------- Plot Trajectories ---------------------------- #
p_hat_traj_CL = getindex.(theta_hat_traj_CL, 1)
q_hat_traj_CL = getindex.(theta_hat_traj_CL, 2)

# Plot trajectories
CL_plot = plot(p_hat_traj_CL, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:right, legendfont=font(15), guidefont=font(20), xtickfont=font(15), ytickfont=font(15));
plot!(CL_plot, q_hat_traj_CL, label = "q");
# Add points for clarity
# scatter!(CL_plot, p_hat_traj_CL, label = nothing);
# scatter!(CL_plot, q_hat_traj_CL, label = nothing);
# Add horizontal line for MLEs
hline!(CL_plot, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash);
# Add vertical line for end of pilot study
vline!(CL_plot, [K_max], label = nothing, linewidth=1, linecolor=:black)
savefig(CL_plot, plotsdir("Blood_Type", "Chan_Ledolter_Traj.pdf"))


# Compute bounds of pointwise 95% confidence band for log-likelihood ratio
# Note: SE changes after end of pilot study. For nicer plotting, we include the transition point in both regions
wald_mult = quantile(Normal(), 0.975)
ucls_pilot = deepcopy(lik_rat_traj_CL[1:K_max]) .+ wald_mult .* SE_pilot
lcls_pilot = deepcopy(lik_rat_traj_CL[1:K_max]) .- wald_mult .* SE_pilot
ucls_final = deepcopy(lik_rat_traj_CL[K_max:end]) .+ wald_mult .* SE_final
lcls_final = deepcopy(lik_rat_traj_CL[K_max:end]) .- wald_mult .* SE_final
ucls_CL = [ucls_pilot; ucls_final]
lcls_CL = [lcls_pilot; lcls_final]


pilot_iterations = collect(eachindex(ucls_pilot))
final_iterations = collect(K_max:K_max+length(ucls_final)-1)
CI_iterations = [pilot_iterations; final_iterations]



CL_lik_rat_plot = plot(lik_rat_traj_CL, label = "Estimate", xlabel = "Iteration", ylabel = "Log-Likelihood Ratio", size=(800, 600), margin=10mm, legendfont=font(15), guidefont=font(20), xtickfont=font(15), ytickfont=font(15))
vline!(CL_lik_rat_plot, [K_max], label = nothing, linewidth=1, linecolor=:black)
# Add confidence band
plot!(CL_lik_rat_plot, final_iterations, ucls_final, linecolor=:red, linestyle=:dash, label = "95% Confidence Band")
plot!(CL_lik_rat_plot, final_iterations, lcls_final, linecolor=:red, linestyle=:dash, label = nothing)
savefig(CL_lik_rat_plot, plotsdir("Blood_Type", "Chan_Ledolter_lik_ratio_Traj.pdf"))

CI_lik_rat_plot_zoomed = plot(final_iterations, lik_rat_traj_CL[K_max:end], label = "Estimate", xlabel = "Iteration", ylabel = "Log-Likelihood Ratio", size=(800, 600), margin=10mm, legendfont=font(15), guidefont=font(20), xtickfont=font(15), ytickfont=font(15))
vline!(CI_lik_rat_plot_zoomed, [K_max], label = nothing, linewidth=1, linecolor=:black)
plot!(CI_lik_rat_plot_zoomed, final_iterations, ucls_final, linecolor=:red, linestyle=:dash, label = "95% Confidence Band")
plot!(CI_lik_rat_plot_zoomed, final_iterations, lcls_final, linecolor=:red, linestyle=:dash, label = nothing)
savefig(CI_lik_rat_plot_zoomed, plotsdir("Blood_Type", "Chan_Ledolter_lik_ratio_Traj_zoomed.pdf"))


# ---------------------------------------------------------------------------- #
#                             Booth and Hobert MCEM                            #
# ---------------------------------------------------------------------------- #

COUNTERS["one_X_given_Y_iid"] = 0
Random.seed!(3)


# # Default values for my implementation
# alpha = 0.25    # Confidence level for building intervals used to check for augmenting MC size
# k = 3   # Fraction by which to augment the MC sample size when necessary
# tau = 0.002   # Relative error threshold for terminating.
# delta = 0.001   # Additive constant for denominator of relative error
# M_init = 10  # Initial MC sample size

# Run algorithm
theta_hat_traj_BH , n_iter_BH, M_traj_BH = run_MCEM_Booth_Hobert(theta_init, Y; return_trajectory=true, return_diagnostics=true)
println("Number of MC draws for Booth and Hobert MCEM:  $(COUNTERS["one_X_given_Y_iid"])")

# Extract final estimate
theta_hat_BH = theta_hat_traj_BH[end]

# ----------------------------- Plot trajectories ---------------------------- #
# Note: This plot is a little weird, because I want to include the MC size at each iteration. This is accomplished with the twinx() function. It's not obvious to me which Y-axis to put on which side, so I did it both ways.

p_hat_traj_BH = getindex.(theta_hat_traj_BH, 1)
q_hat_traj_BH = getindex.(theta_hat_traj_BH, 2)

# Estimate on left, MC size on right
BH_plot = plot(p_hat_traj_BH, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:left, legendfont=font(15), guidefont=font(20), xtickfont=font(15), ytickfont=font(15));
plot!(BH_plot, q_hat_traj_BH, label = "q")
plot!(BH_plot, [10000], color=:black, xlims=xlims(BH_plot), ylims=ylims(BH_plot), label = "MC Size", legendfont=font(15))
rhs = twinx()
plot!(rhs, M_traj_BH, label = nothing, color=:black, linetype=:steppost, ylabel="MC Sample Size", guidefont=font(20), ytickfont=font(15))


# MC size on left, estimate on right
BH_plot2 = plot(M_traj_BH, label = nothing, color=:purple, linetype=:steppost, ylabel="MC Sample Size", guidefont=font(20), xlabel = "Iteration", margin=10mm, size=(800, 600), xtickfont=font(15), ytickfont=font(15))
rhs = twinx()
plot!(rhs, p_hat_traj_BH,  label = "p", ylabel = "Estimate", legend=:left, legendfont=font(15), guidefont=font(20), xtickfont=font(15), ytickfont=font(15))
plot!(rhs, q_hat_traj_BH, label = "q")
plot!(rhs, [10000], color=:purple, xlims=xlims(rhs), ylims=ylims(rhs), label = "MC Size", legendfont=font(15))
hline!(rhs, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
savefig(BH_plot2, plotsdir("Blood_Type", "Booth_Hobert_Traj.pdf"))



# ---------------------------------------------------------------------------- #
#                    Analysis: AMCEM (Caffo, Jank and Jones)                   #
# ---------------------------------------------------------------------------- #


COUNTERS["one_X_given_Y_iid"] = 0
Random.seed!(4)

# --------------- Set control parameters for ascent-based MCEM --------------- #
alpha1 = 0.2    # confidence level for checking whether to augment MC sample size
alpha2 = 0.2    # confidence level for computing next step's initial MC sample size
alpha3 = 0.1    # confidence level for checking whether to terminate MCEM
k = 2           # when augmenting MC sample, add M/k new points
atol = 1e-3     # Absolute tolerance for convergence. 

AMCEM_control = Ascent_MCEM_Control(alpha1, alpha2, alpha3, k, atol)
ascent_MCEM_control = AMCEM_control

M_init = 10



theta_AMCEM, all_theta_hat_AMCEMs, M_traj_AMCEM = run_ascent_MCEM([0.3, 0.3], Y, M_init, AMCEM_control; diagnostics=true)
println("Number of MC draws for Caffo et al. MCEM:  $(COUNTERS["one_X_given_Y_iid"])")



p_hat_traj_AMCEM = getindex.(all_theta_hat_AMCEMs, 1)
q_hat_traj_AMCEM = getindex.(all_theta_hat_AMCEMs, 2)



# MC size on left, estimate on right
AMCEM_plot = plot(M_traj_AMCEM, label = nothing, color=:purple, linetype=:steppost, ylabel="MC Sample Size", guidefont=font(20), xlabel = "Iteration", margin=10mm, size=(800, 600), xtickfont=font(15), ytickfont=font(15));
rhs = twinx()
plot!(rhs, p_hat_traj_AMCEM,  label = "p", ylabel = "Estimate", legend=:left, legendfont=font(15), guidefont=font(20), xtickfont=font(16), ytickfont=font(16));
plot!(rhs, q_hat_traj_AMCEM, label = "q");
plot!(rhs, [10000], color=:purple, xlims=xlims(rhs), ylims=ylims(rhs), label = "MC Size", legendfont=font(15));
hline!(rhs, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
savefig(AMCEM_plot, plotsdir("Blood_Type", "AMCEM_Traj.pdf"))





# ---------------------------------------------------------------------------- #
#                Stochastic Approximation EM: Objective Function               #
# ---------------------------------------------------------------------------- #


COUNTERS["one_X_given_Y_iid"] = 0
Random.seed!(5)

# --------------- Set control parameters for ascent-based MCEM --------------- #
M_SAEM = 10      # MC size for each iteration of SAEM
SA_rate = 0.7     # Power on 1/k for step size
# SA_rate = 1.0     # Power on 1/k for step size
# B_SAEM = 200         # Number of SAEM iterations
B_SAEM = 50         # Number of SAEM iterations


all_theta_hat_SAEMs = run_SAEM(theta_init, Y, M_SAEM, SA_rate, B_SAEM)
println("Number of MC draws for Objective Function SAEM:  $(COUNTERS["one_X_given_Y_iid"])")



p_hat_traj_SAEM = getindex.(all_theta_hat_SAEMs, 1)
q_hat_traj_SAEM = getindex.(all_theta_hat_SAEMs, 2)



# Plot trajectories
SAEM_plot = plot(p_hat_traj_SAEM, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:right, legendfont=font(15), guidefont=font(20), xtickfont=font(15), ytickfont=font(15));
plot!(SAEM_plot, q_hat_traj_SAEM, label = "q");
# Add horizontal line for MLEs
hline!(SAEM_plot, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
savefig(SAEM_plot, plotsdir("Blood_Type", "SAEM_Traj_obj_fun.pdf"))





# MC size on left, estimate on right
SAEM_plot2 = plot(M_SAEM .* (1:B_SAEM), label = nothing, color=:purple, linetype=:steppost, ylabel="MC Sample Size", guidefont=font(20), xlabel = "Iteration", margin=10mm, size=(800, 600), xtickfont=font(15), ytickfont=font(15));
rhs = twinx()
plot!(rhs, p_hat_traj_SAEM,  label = "p", ylabel = "Estimate", legend=:left, legendfont=font(15), guidefont=font(20), xtickfont=font(16), ytickfont=font(16));
plot!(rhs, q_hat_traj_SAEM, label = "q");
plot!(rhs, [10000], color=:purple, xlims=xlims(rhs), ylims=ylims(rhs), label = "MC Size", legendfont=font(15));
hline!(rhs, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
savefig(SAEM_plot2, plotsdir("Blood_Type", "SAEM_Traj_obj_fun_MC_size.pdf"))




# ---------------------------------------------------------------------------- #
#                  Stochastic Approximation EM: Obs Data Score                 #
# ---------------------------------------------------------------------------- #


COUNTERS["one_X_given_Y_iid"] = 0
Random.seed!(6)

# --------------- Set control parameters for ascent-based MCEM --------------- #
M_SAEM = 10      # MC size for each iteration of SAEM
SA_rate = 0.7     # Power on 1/k for step size
# SA_rate = 1.0     # Power on 1/k for step size
# B_SAEM = 200         # Number of SAEM iterations
B_SAEM = 50         # Number of SAEM iterations


all_theta_hat_SAEMs_score = run_SAEM_score(theta_init, Y, M_SAEM, SA_rate, B_SAEM)
println("Number of MC draws for Score SAEM:  $(COUNTERS["one_X_given_Y_iid"])")



p_hat_traj_SAEM_score = getindex.(all_theta_hat_SAEMs_score, 1)
q_hat_traj_SAEM_score = getindex.(all_theta_hat_SAEMs_score, 2)



# Plot trajectories
SAEM_score_plot = plot(p_hat_traj_SAEM_score, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legendfont=font(15), guidefont=font(20), xtickfont=font(15), ytickfont=font(15));
plot!(SAEM_score_plot, q_hat_traj_SAEM_score, label = "q");
# Add horizontal line for MLEs
hline!(SAEM_score_plot, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
savefig(SAEM_score_plot, plotsdir("Blood_Type", "SAEM_Traj_score.pdf"))





# MC size on left, estimate on right
SAEM_score_plot2 = plot(M_SAEM .* (1:B_SAEM), label = nothing, color=:purple, linetype=:steppost, ylabel="MC Sample Size", guidefont=font(20), xlabel = "Iteration", margin=10mm, size=(800, 600), xtickfont=font(15), ytickfont=font(15));
rhs = twinx()
plot!(rhs, p_hat_traj_SAEM_score,  label = "p", ylabel = "Estimate", legend=:top, legendfont=font(15), guidefont=font(20), xtickfont=font(16), ytickfont=font(16));
plot!(rhs, q_hat_traj_SAEM_score, label = "q");
plot!(rhs, [10000], color=:purple, xlims=xlims(rhs), ylims=ylims(rhs), label = "MC Size", legendfont=font(15));
hline!(rhs, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
savefig(SAEM_plot2, plotsdir("Blood_Type", "SAEM_Traj_score_MC_size.pdf"))



# ---------------------------------------------------------------------------- #
#                                SAEM Averaging                                #
# ---------------------------------------------------------------------------- #

# ------------------------ Compute cumulative averages ----------------------- #
p_cum_mean_SAEM_obj = [sum(p_hat_traj_SAEM[1:n]) / n for n in 1:length(p_hat_traj_SAEM)]
q_cum_mean_SAEM_obj = [sum(q_hat_traj_SAEM[1:n]) / n for n in 1:length(q_hat_traj_SAEM)]

p_cum_mean_SAEM_score = [sum(p_hat_traj_SAEM_score[1:n]) / n for n in 1:length(p_hat_traj_SAEM_score)]
q_cum_mean_SAEM_score = [sum(q_hat_traj_SAEM_score[1:n]) / n for n in 1:length(q_hat_traj_SAEM_score)]


# ------------------------- Plot cumulative averages ------------------------- #
SAEM_obj_mean_plot = deepcopy(SAEM_plot);
plot!(SAEM_obj_mean_plot, p_cum_mean_SAEM_obj, label = nothing, color=1, linestyle=:dash);
plot!(SAEM_obj_mean_plot, q_cum_mean_SAEM_obj, label = nothing, color=:red, linestyle=:dash);
hline!(SAEM_obj_mean_plot, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
savefig(SAEM_obj_mean_plot, plotsdir("Blood_Type", "SAEM_Mean_Traj_obj.pdf"))



SAEM_score_mean_plot = deepcopy(SAEM_score_plot);
plot!(SAEM_score_mean_plot, p_cum_mean_SAEM_score, label = nothing, color=1, linestyle=:dash);
plot!(SAEM_score_mean_plot, q_cum_mean_SAEM_score, label = nothing, color=:red, linestyle=:dash);
hline!(SAEM_score_mean_plot, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
savefig(SAEM_score_mean_plot, plotsdir("Blood_Type", "SAEM_Mean_Traj_score.pdf"))



# -------------------------- Compute moving averages ------------------------- #
window_size = Integer(ceil(sqrt(B_SAEM)))

p_win_mean_SAEM_obj = [sum(p_hat_traj_SAEM[(i+1):(i+window_size)]) / window_size for i in 0:(length(p_hat_traj_SAEM)-window_size)]
q_win_mean_SAEM_obj = [sum(q_hat_traj_SAEM[(i+1):(i+window_size)]) / window_size for i in 0:(length(q_hat_traj_SAEM)-window_size)]

p_win_mean_SAEM_score = [sum(p_hat_traj_SAEM_score[(i+1):(i+window_size)]) / window_size for i in 0:(length(p_hat_traj_SAEM_score)-window_size)]
q_win_mean_SAEM_score = [sum(q_hat_traj_SAEM_score[(i+1):(i+window_size)]) / window_size for i in 0:(length(q_hat_traj_SAEM_score)-window_size)]


# --------------------------- Plot moving averages --------------------------- #
SAEM_obj_window_mean_plot = deepcopy(SAEM_plot);
plot!(SAEM_obj_window_mean_plot, (window_size):B_SAEM, p_win_mean_SAEM_obj, label = nothing, color=1, linestyle=:dash);
plot!(SAEM_obj_window_mean_plot, (window_size):B_SAEM, q_win_mean_SAEM_obj, label = nothing, color=:red, linestyle=:dash);
hline!(SAEM_obj_window_mean_plot, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)

SAEM_score_window_mean_plot = deepcopy(SAEM_plot);
plot!(SAEM_score_window_mean_plot, (window_size):B_SAEM, p_win_mean_SAEM_score, label = nothing, color=1, linestyle=:dash);
plot!(SAEM_score_window_mean_plot, (window_size):B_SAEM, q_win_mean_SAEM_score, label = nothing, color=:red, linestyle=:dash);
hline!(SAEM_score_window_mean_plot, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)




# ---------------------------------------------------------------------------- #
#                        Monte Carlo Maximum Likelihood                        #
# ---------------------------------------------------------------------------- #



COUNTERS["one_X_given_Y_iid"] = 0
Random.seed!(7)

# ---------------------- Set control parameters for MCML --------------------- #
M_MCML = 1000


# --------------------------------- Run MCML --------------------------------- #
theta_hat_MCML = run_MCML(theta_init, Y, M_MCML)
println("Number of MC draws for MCML:  $(COUNTERS["one_X_given_Y_iid"])")
theta_hat_MCML2 = run_MCML(theta_hat_MCML, Y, M_MCML)



# ---------------------------------------------------------------------------- #
#                          Monte Carlo Newton-Raphson                          #
# ---------------------------------------------------------------------------- #

#! I'm cutting this. My naive implementation doesn't respect the summation constraint for the probabilities and it's just not worth fixing.


COUNTERS["one_X_given_Y_iid"] = 0
Random.seed!(8)


# ---------------------- Set control parameters for MCNR --------------------- #
M_MCNR = 1000
alpha_MCNR = 0.1


# --------------------------------- Run MCNR --------------------------------- #
all_theta_hat_MCNRs = run_MCNR(theta_init, Y, M_MCNR, alpha=alpha_MCNR, return_traj=true)
println("Number of MC draws for Score SAEM:  $(COUNTERS["one_X_given_Y_iid"])")



p_hat_traj_SAEM_score = getindex.(all_theta_hat_SAEMs_score, 1)
q_hat_traj_SAEM_score = getindex.(all_theta_hat_SAEMs_score, 2)



# Plot trajectories
SAEM_score_plot = plot(p_hat_traj_SAEM_score, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legendfont=font(15), guidefont=font(20), xtickfont=font(15), ytickfont=font(15));
plot!(SAEM_score_plot, q_hat_traj_SAEM_score, label = "q");
# Add horizontal line for MLEs
hline!(SAEM_score_plot, [theta_MLE], label = nothing, linewidth=1, linecolor=:black, linestyle=:dash)
savefig(SAEM_score_plot, plotsdir("Blood_Type", "SAEM_Traj_score.pdf"))

