
using Pkg
using DrWatson

push!(LOAD_PATH, srcdir("SLR_Example", "src"))

using SLR_Example


using Random
using Distributions
using LinearAlgebra
using ProgressMeter

using Plots

# Run this before testing to make sure that Julia looks in the right place for tests
Pkg.activate(srcdir("SLR_Example"))

# Run this to edit the packages used for testing
# Pkg.activate(srcdir("SLR_Example", "test"))


# ---------------------------------------------------------------------------- #
#                              Construct Some Data                             #
# ---------------------------------------------------------------------------- #

Random.seed!(1)

p_0 = 0.3
q_0 = 0.4
r_0 = 1 - p_0 - q_0
theta_0 = [p_0, q_0]

prob_vec = get_cell_probs(theta_0)

n = 1000
Y = rand(Multinomial(n, prob_vec), 1)
1


theta1 = [1/3, 1/3]
theta = theta1




beta_0 = 1.0
mu_0 = 1.0
tau_0 = 1.0
sigma_0 = 1.0
theta_0 = [beta_0, sigma_0]

n = 100
X = rand(Normal(mu_0, tau_0), n)
epsilon = rand(Normal(0, sigma_0), n)
Y = beta_0 * X + epsilon

x = X[1]
y = Y[1]


theta1 = [1.0, 1.0]
theta = theta1
theta2 = [2.0, 1.0]
theta3 = [1.0, 2.0]
theta4 = [2.0, 2.0]
all_thetas = [theta1, theta2, theta3, theta4]
theta_fixed = [mu_0, tau_0]



# ---------------------------------------------------------------------------- #
#                                 Analysis: MLE                                #
# ---------------------------------------------------------------------------- #

theta_MLE = obs_data_MLE(Y, theta_fixed)
SE_MLE = obs_data_MLE_Cov(theta_MLE, Y, theta_fixed)



# ---------------------------------------------------------------------------- #
#                                 Analysis: EM                                 #
# ---------------------------------------------------------------------------- #

theta_EM = run_EM(theta4, Y, theta_fixed)
SE_EM = EM_COV_formula(theta_EM, Y, theta_fixed)


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
