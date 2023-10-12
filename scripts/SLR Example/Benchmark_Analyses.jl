
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
p_hat_MLE, q_hat_MLE = theta_MLE
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
M_small = 100
M_large = 1000

# Preliminary number of iterations
K_small = 50
K_large = 20

theta_hat_MCEM1, theta_hat_traj_MCEM1 = run_MCEM_fixed_iteration_count(theta_init, Y, M, K; return_trajectory=true)
println("Number of MC draws for Naive MCEM:  $(COUNTERS["one_X_given_Y_iid"])")

function run_both_MCEM_fixed_iteration_count(theta_init, Y, M_small, M_large, K_small, K_large)
    theta_hat_MCEM_small = run_MCEM_fixed_iteration_count(theta_init, Y, M_small, K_small; return_trajectory=false)
    theta_hat_MCEM_large = run_MCEM_fixed_iteration_count(theta_hat_MCEM_small, Y, M_large, K_large; return_trajectory=false)
    return theta_hat_MCEM_large
end

time_naive_MCEM = @benchmark run_both_MCEM_fixed_iteration_count($theta_init, $Y, $M_small, $M_large, $K_small, $K_large) samples=num_reps_time


COUNTERS["one_X_given_Y_iid"] = 0
all_theta_hats_MCEM_fixed = [run_MCEM_fixed_iteration_count(theta_init, Y, M, K) for i in 1:num_reps_time]
println("Number of MC draws for $num_reps_time replicates of Naive MCEM:  $(COUNTERS["one_X_given_Y_iid"])")

# Compute mean relative error of estimates
p_hats_MCEM_fixed = getindex.(all_theta_hats_MCEM_fixed, 1);
all_p_errs_MCEM_fixed = abs.(p_hats_MCEM_fixed .- p_hat_MLE) ./ p_hat_MLE;
mean_p_err_MCEM_fixed = mean(all_p_errs_MCEM_fixed)

q_hats_MCEM_fixed = getindex.(all_theta_hats_MCEM_fixed, 2);
all_q_errs_MCEM_fixed = abs.(q_hats_MCEM_fixed .- q_hat_MLE) ./ q_hat_MLE;
mean_q_err_MCEM_fixed = mean(all_q_errs_MCEM_fixed)




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


time_MCEM_CL = @benchmark run_MCEM_Chan_Ledolter($Y, $theta_init, $M_pilot, $K_max, $R_keep, $B, $delta) samples=num_reps_time


COUNTERS["one_X_given_Y_iid"] = 0
all_theta_hats_MCEM_CL = []
for _ in 1:num_reps_time
    theta_hat_traj_CL, _, _, _ = run_MCEM_Chan_Ledolter(Y, theta_init, M_pilot, K_max, R_keep, B, delta; diagnostics=true)
    push!(all_theta_hats_MCEM_CL, theta_hat_traj_CL[end])
end
println("Number of MC draws for $num_reps_time replicates of Chan and Ledolter MCEM:  $(COUNTERS["one_X_given_Y_iid"])")


# Compute mean relative error of estimates
p_hats_MCEM_CL = getindex.(all_theta_hats_MCEM_CL, 1);
all_p_errs_MCEM_CL = abs.(p_hats_MCEM_CL .- p_hat_MLE) ./ p_hat_MLE;
mean_p_err_MCEM_CL = mean(all_p_errs_MCEM_CL)

q_hats_MCEM_CL = getindex.(all_theta_hats_MCEM_CL, 2);
all_q_errs_MCEM_CL = abs.(q_hats_MCEM_CL .- q_hat_MLE) ./ q_hat_MLE;
mean_q_err_MCEM_CL = mean(all_q_errs_MCEM_CL)

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


time_MCEM_BH = @benchmark run_MCEM_Booth_Hobert($theta_init, $Y; return_trajectory=true, return_diagnostics=true) samples=num_reps_time


COUNTERS["one_X_given_Y_iid"] = 0
all_theta_hats_MCEM_CL = []
for _ in 1:num_reps_time
    theta_hat_BH = run_MCEM_Booth_Hobert(theta_init, Y; return_trajectory=false, return_diagnostics=false)
    push!(all_theta_hats_MCEM_CL, theta_hat_BH)
end
println("Number of MC draws for $num_reps_time replicates of Booth and Hobert MCEM:  $(COUNTERS["one_X_given_Y_iid"])")


# Compute mean relative error of estimates
p_hats_MCEM_BH = getindex.(all_theta_hats_MCEM_CL, 1);
all_p_errs_MCEM_BH = abs.(p_hats_MCEM_BH .- p_hat_MLE) ./ p_hat_MLE;
mean_p_err_MCEM_BH = mean(all_p_errs_MCEM_BH)

q_hats_MCEM_BH = getindex.(all_theta_hats_MCEM_CL, 2);
all_q_errs_MCEM_BH = abs.(q_hats_MCEM_BH .- q_hat_MLE) ./ q_hat_MLE;
mean_q_err_MCEM_BH = mean(all_q_errs_MCEM_BH)



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


time_AMCEM = @benchmark run_ascent_MCEM($theta_init, $Y, $M_init, $AMCEM_control; diagnostics=true) samples=num_reps_time seconds=60
time_AMCEM


COUNTERS["one_X_given_Y_iid"] = 0
all_theta_hats_AMCEM = []
for _ in 1:num_reps_time
    theta_hat_AMCEM = run_ascent_MCEM(theta_init, Y, M_init, AMCEM_control; diagnostics=false)
    push!(all_theta_hats_AMCEM, theta_hat_AMCEM)
end
println("Number of MC draws for $num_reps_time replicates of Caffo et al. MCEM:  $(COUNTERS["one_X_given_Y_iid"])")


# Compute mean relative error of estimates
p_hats_AMCEM = getindex.(all_theta_hats_AMCEM, 1);
all_p_errs_AMCEM = abs.(p_hats_AMCEM .- p_hat_MLE) ./ p_hat_MLE;
mean_p_err_AMCEM = mean(all_p_errs_AMCEM)

q_hats_AMCEM = getindex.(all_theta_hats_AMCEM, 2);
all_q_errs_AMCEM = abs.(q_hats_AMCEM .- q_hat_MLE) ./ q_hat_MLE;
mean_q_err_AMCEM = mean(all_q_errs_AMCEM)








# ---------------------------------------------------------------------------- #
#                Stochastic Approximation EM: Objective Function               #
# ---------------------------------------------------------------------------- #


COUNTERS["one_X_given_Y_iid"] = 0
Random.seed!(5)

# --------------- Set control parameters for ascent-based MCEM --------------- #
M_SAEM = 10      # MC size for each iteration of SAEM
SA_rate = 1.0     # Power on 1/k for step size
B_SAEM = 50         # Number of SAEM iterations


all_theta_hat_SAEMs = run_SAEM(theta_init, Y, M_SAEM, SA_rate, B_SAEM)
println("Number of MC draws for Objective Function SAEM:  $(COUNTERS["one_X_given_Y_iid"])")


time_SAEM = @benchmark run_SAEM($theta_init, $Y, $M_SAEM, $SA_rate, $B_SAEM) samples=num_reps_time

COUNTERS["one_X_given_Y_iid"] = 0
all_theta_hats_SAEM = []
for _ in 1:num_reps_time
    theta_hat_SAEM = run_SAEM(theta_init, Y, M_SAEM, SA_rate, B_SAEM)
    push!(all_theta_hats_SAEM, theta_hat_SAEM[end])
end
println("Number of MC draws for $num_reps_time replicates of Objective Function SAEM:  $(COUNTERS["one_X_given_Y_iid"])")


# Compute mean relative error of estimates
p_hats_SAEM = getindex.(all_theta_hats_SAEM, 1);
all_p_errs_SAEM = abs.(p_hats_SAEM .- p_hat_MLE) ./ p_hat_MLE;
mean_p_err_SAEM = mean(all_p_errs_SAEM)

q_hats_SAEM = getindex.(all_theta_hats_SAEM, 2);
all_q_errs_SAEM = abs.(q_hats_SAEM .- q_hat_MLE) ./ q_hat_MLE;
mean_q_err_SAEM = mean(all_q_errs_SAEM)


# ---------------------------------------------------------------------------- #
#                  Stochastic Approximation EM: Obs Data Score                 #
# ---------------------------------------------------------------------------- #


COUNTERS["one_X_given_Y_iid"] = 0
Random.seed!(6)

# --------------- Set control parameters for ascent-based MCEM --------------- #
M_SAEM = 10      # MC size for each iteration of SAEM
SA_rate = 1.0     # Power on 1/k for step size
B_SAEM = 50         # Number of SAEM iterations


all_theta_hat_SAEMs_score = run_SAEM_score(theta_init, Y, M_SAEM, SA_rate, B_SAEM)
println("Number of MC draws for Score SAEM:  $(COUNTERS["one_X_given_Y_iid"])")


time_SAEM_score = @benchmark run_SAEM_score($theta_init, $Y, $M_SAEM, $SA_rate, $B_SAEM) samples=num_reps_time


COUNTERS["one_X_given_Y_iid"] = 0
all_theta_hats_SAEM_score = []
for _ in 1:num_reps_time
    theta_hat_SAEM_score = run_SAEM_score(theta_init, Y, M_SAEM, SA_rate, B_SAEM)
    push!(all_theta_hats_SAEM_score, theta_hat_SAEM_score[end])
end
println("Number of MC draws for $num_reps_time replicates of Score SAEM:  $(COUNTERS["one_X_given_Y_iid"])")



# Compute mean relative error of estimates
p_hats_SAEM_score = getindex.(all_theta_hats_SAEM_score, 1);
all_p_errs_SAEM_score = abs.(p_hats_SAEM_score .- p_hat_MLE) ./ p_hat_MLE;
mean_p_err_SAEM_score = mean(all_p_errs_SAEM_score)

q_hats_SAEM_score = getindex.(all_theta_hats_SAEM_score, 2);
all_q_errs_SAEM_score = abs.(q_hats_SAEM_score .- q_hat_MLE) ./ q_hat_MLE;
mean_q_err_SAEM_score = mean(all_q_errs_SAEM_score)



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


time_MCML = @benchmark run_MCML($theta_init, $Y, $M_MCML) samples=num_reps_time


COUNTERS["one_X_given_Y_iid"] = 0
all_theta_hats_MCML = []
for _ in 1:num_reps_time
    theta_hat_MCML = run_MCML(theta_init, Y, M_MCML)
    push!(all_theta_hats_MCML, theta_hat_MCML)
end
println("Number of MC draws for $num_reps_time replicates of MCML:  $(COUNTERS["one_X_given_Y_iid"])")


# Compute mean relative error of estimates
p_hats_MCML = getindex.(all_theta_hats_MCML, 1);
all_p_errs_MCML = abs.(p_hats_MCML .- p_hat_MLE) ./ p_hat_MLE;
mean_p_err_MCML = mean(all_p_errs_MCML)

q_hats_MCML = getindex.(all_theta_hats_MCML, 2);
all_q_errs_MCML = abs.(q_hats_MCML .- q_hat_MLE) ./ q_hat_MLE;
mean_q_err_MCML = mean(all_q_errs_MCML)




# -------------------------- Two iterations of MCML -------------------------- #

Random.seed!(8)

function run_double_MCML(theta_init, Y, M_MCML)
    theta_hat_MCML = run_MCML(theta_init, Y, M_MCML)
    theta_hat_MCML2 = run_MCML(theta_hat_MCML, Y, M_MCML)
    return theta_hat_MCML2
end

time_double_MCML = @benchmark run_double_MCML($theta_init, $Y, $M_MCML) samples=num_reps_time


COUNTERS["one_X_given_Y_iid"] = 0
all_theta_hats_double_MCML = []
for _ in 1:num_reps_time
    theta_hat_double_MCML = run_double_MCML(theta_init, Y, M_MCML)
    push!(all_theta_hats_double_MCML, theta_hat_double_MCML)
end
println("Number of MC draws for $num_reps_time replicates of double MCML:  $(COUNTERS["one_X_given_Y_iid"])")


# Compute mean relative error of estimates
p_hats_double_MCML = getindex.(all_theta_hats_double_MCML, 1);
all_p_errs_double_MCML = abs.(p_hats_double_MCML .- p_hat_MLE) ./ p_hat_MLE;
mean_p_err_double_MCML = mean(all_p_errs_double_MCML)

q_hats_double_MCML = getindex.(all_theta_hats_double_MCML, 2);
all_q_errs_double_MCML = abs.(q_hats_double_MCML .- q_hat_MLE) ./ q_hat_MLE;
mean_q_err_double_MCML = mean(all_q_errs_double_MCML)