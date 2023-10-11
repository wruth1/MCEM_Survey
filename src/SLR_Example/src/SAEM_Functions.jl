

export SA_step_size, get_zeta, get_zeta_list
export MC_num_O_alleles, MC_num_A_alleles, MC_num_B_alleles
export SAEM_p_hat, SAEM_q_hat, SAEM_theta_hat
export one_MCEM_iteration, run_SAEM

export run_SAEM_score





# ---------------------------------------------------------------------------- #
#                       SAEM: Target = Objective Function                      #
# ---------------------------------------------------------------------------- #


"""
Weight on the new sample at each iteration.
"""
SA_step_size(k, SA_rate) = k^(-SA_rate)


"""
Compute the SA weights for each sample at iteration k.
"""
function get_zeta_list(k, SA_rate)
    all_gammas = [SA_step_size(j, SA_rate) for j in 1:k]
    all_zetas = []
    for j in 1:k
        this_zeta = all_gammas[j]
        for i in (j+1):k    # Empty if j+1 > k
            this_zeta *= (1 - all_gammas[i])
        end
        push!(all_zetas, this_zeta)
    end
    return all_zetas
end





# ------------ Count number of alleles of each type in a MC sample ----------- #

function MC_num_O_alleles(all_Xs)
    all_num_Os = [num_O_alleles(all_Xs[i]) for i in 1:length(all_Xs)]
    return sum(all_num_Os)
end

function MC_num_A_alleles(all_Xs)
    all_num_As = [num_A_alleles(all_Xs[i]) for i in 1:length(all_Xs)]
    return sum(all_num_As)
end

function MC_num_B_alleles(all_Xs)
    all_num_Bs = [num_B_alleles(all_Xs[i]) for i in 1:length(all_Xs)]
    return sum(all_num_Bs)
end



# ---------------- Estimate p and q at current SAEM iteration ---------------- #

"""
Compute SAEM estimate of p for given list of number of A alleles (na_list), list of weights (zeta_list), and total number of people (n).
"""
function SAEM_p_hat(na_list, zeta_list, n)
    Na = sum(na_list .* zeta_list)
    N_den = 2 * n * sum(zeta_list)

    return Na / N_den
end


"""
Compute SAEM estimate of q for given list of number of B alleles (nb_list), list of weights (zeta_list), and total number of people (n).
"""
function SAEM_q_hat(nb_list, zeta_list, n)
    Nb = sum(nb_list .* zeta_list)
    N_den = 2 * n * sum(zeta_list)

    return Nb / N_den
end


"""
Estimate parameters given the provided SAEM history.
"""
function SAEM_theta_hat(na_list, nb_list, zeta_list, n)
    p_hat = SAEM_p_hat(na_list, zeta_list, n)
    q_hat = SAEM_q_hat(nb_list, zeta_list, n)
    return p_hat, q_hat
end



# --------------------------------- Run SAEM --------------------------------- #

"""
One iteration of SAEM.
theta_old: current estimate of theta
Y: vector of observed phenotypes
M: number of MC samples to draw
SA_rate: rate of decay of SA weights
na_list: list of number of A alleles at each previous iteration
nb_list: list of number of B alleles at each previous iteration
"""
function one_MCEM_iteration(theta_old, Y, M, SA_rate, na_list, nb_list)
    k = length(na_list) + 1     # Iteration number
    n = sum(Y)                  # Total number of people
    
    all_Xs = sample_X_given_Y_iid(M, theta_old, Y)
    na = MC_num_A_alleles(all_Xs)
    nb = MC_num_B_alleles(all_Xs)
    zeta_list = get_zeta_list(k, SA_rate)
    push!(na_list, na)
    push!(nb_list, nb)

    theta_new = SAEM_theta_hat(na_list, nb_list, zeta_list, n * M)
    return theta_new, na_list, nb_list
end


"""
Run SAEM for B iterations.
    theta_init: initial estimate of theta
    Y: vector of observed phenotypes
    M: number of MC samples to draw at each iteration
    SA_rate: rate of decay of SA weights
    B: number of iterations to run
"""
function run_SAEM(theta_init, Y, M, SA_rate, B)
    na_list = []
    nb_list = []
    theta_hat_list = []
    theta_hat = theta_init

    for _ in 1:B
        theta_hat, na_list, nb_list = one_MCEM_iteration(theta_hat, Y, M, SA_rate, na_list, nb_list)
        push!(theta_hat_list, theta_hat)
    end

    return theta_hat_list
end







# ---------------------------------------------------------------------------- #
#                            SAEM for obs data score                           #
# ---------------------------------------------------------------------------- #


# -------------- Reparameterize problem using logits of p and q -------------- #
# ------- Write p1 = logit(p), q1 = logit(q) and theta1 = logit.(theta) ------ #
# ------- Note: logit() is built-in to Julia. Its inverse is logistic() ------ #

"""
Compute gradient of complete data log-likelihood with parameters on logit-scale.
"""
function complete_data_score_logit(theta1, Y, X)
    theta = logistic.(theta1)
    raw_score = complete_data_score(theta, Y, X)
    jacobian = diagm(theta .* (1 .- theta))
    return jacobian * raw_score
end

"""
Perform an SA step on the logit-scale.
Requires MC sampled Xs to be passed as an argument (i.e. they must already have been generated).
"""
function one_MCEM_iteration_score_logit(theta1_old, Y, all_Xs, k, SA_rate)
    
    all_scores = [complete_data_score_logit(theta1_old, Y, all_Xs[i]) for i in 1:M]
    mean_score = mean(all_scores)

    step_size = SA_step_size(k, SA_rate)
    theta1_new = theta1_old + step_size * mean_score
    return theta1_new
end





# --------------------------------- Run SAEM --------------------------------- #

"""
One iteration of SAEM. Convert theta to logit-scale for update, then convert back.
theta_old: current estimate of theta
Y: vector of observed phenotypes
M: number of MC samples to draw
SA_rate: rate of decay of SA weights
"""
function one_MCEM_iteration_score(theta_old, Y, M, k, SA_rate)
    
    all_Xs = sample_X_given_Y_iid(M, theta_old, Y)
    
    theta1_old = logit.(theta_old)
    theta1_new = one_MCEM_iteration_score_logit(theta1_old, Y, all_Xs, k, SA_rate)
    theta_new = logistic.(theta1_new)

    return theta_new
end


"""
Run SAEM for B iterations.
    theta_init: initial estimate of theta
    Y: vector of observed phenotypes
    M: number of MC samples to draw at each iteration
    SA_rate: rate of decay of SA weights
    B: number of iterations to run
"""
function run_SAEM_score(theta_init, Y, M, SA_rate, B)
    theta_hat_list = []
    theta_hat = theta_init

    for k in 1:B
        theta_hat = one_MCEM_iteration_score(theta_hat, Y, M, k, SA_rate)
        push!(theta_hat_list, theta_hat)
    end

    return theta_hat_list
end