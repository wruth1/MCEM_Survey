

export one_X_given_Y_iid, sample_X_given_Y_iid
export MCEM_update_iid

export conditional_complete_information_iid, conditional_complete_sq_score_iid, conditional_complete_score_cov_iid
export obs_information_formula_iid, MCEM_cov_formula_iid

export Q_MCEM_iid, Q_MCEM_increment_iid, grad_MCEM_iid

export run_MCEM_fixed_iteration_count




# ---------------------------------------------------------------------------- #
#                                 IID Sampling                                 #
# ---------------------------------------------------------------------------- #

# Here, we sample directly from the conditional distribution of X given Y. No proposal distribution, and no importance weights. Should be easy-peasy lemon squeezy.

"""
Genetate a single observation from the conditional distribution of X given Y.
"""
@counted function one_X_given_Y_iid(theta, Y)
    y1, y2, y3, y4 = Y

    alpha1 = get_alpha1(theta)  # P(X2|Y)
    beta1 = get_beta1(theta)    # P(X4|Y)

    x1 = y1
    x6 = y4

    x2_dist = Binomial(y2, alpha1)
    x2 = rand(x2_dist)
    x3 = y2 - x2

    x4_dist = Binomial(y3, beta1)
    x4 = rand(x4_dist)
    x5 = y3 - x4

    this_X = [x1, x2, x3, x4, x5, x6]
    return this_X
end


"""
Generate M observations from the conditional distribution of X given Y.
"""
function sample_X_given_Y_iid(M, theta, Y)
    return [one_X_given_Y_iid(theta, Y) for _ in 1:M]
end

# all_Xs = sample_X_given_Y_iid(100, theta, Y)




"""
Compute the next estimate of theta using MCEM.
MC sample is provided as an argument.
"""
function MCEM_update_iid(Y, all_Xs)

    X_means = []
    for i in eachindex(all_Xs[1])
        this_X_sample = [all_Xs[j][i] for j in eachindex(all_Xs)]
        this_mean = mean(this_X_sample)
        push!(X_means, this_mean)
    end

    theta_hat = complete_data_MLE(Y, X_means)
    return theta_hat
end

"""
Compute the next estimate of theta using MCEM.
MC sample is generated internally and optionally returned.
"""
function MCEM_update_iid(theta_old, Y, M; return_X=false)
    all_Xs = sample_X_given_Y_iid(M, theta_old, Y)
    theta_hat = MCEM_update_iid(Y, all_Xs)

    if !return_X
        return theta_hat
    else
        return theta_hat, all_Xs
    end
end




# ---------------------------------------------------------------------------- #
#                                 SE Estimation                                #
# ---------------------------------------------------------------------------- #

"""
Empirical conditional expectation of complete data information matrix (i.e. negative Hessian of log-likelihood).
MC sample is provided.
"""
function conditional_complete_information_iid(theta, Y, all_Xs)
    sum_Hessian = zeros(2,2)

    for X in all_Xs
        this_Hessian = complete_data_Hessian(theta, Y, X)
        sum_Hessian += this_Hessian
    end

    mean_Hessian = sum_Hessian / length(all_Xs)
    return -mean_Hessian
end


"""
Empirical conditional expectation of complete data information matrix (i.e. negative Hessian of log-likelihood).
MC sample is generated and optionally returned. return_X flags whether to return the MC sample. This argument is not optional to avoid overloading the 3-argument version of this function.
"""
function conditional_complete_information_iid(theta, Y, M, return_X)
    all_Xs = sample_X_given_Y_iid(M, theta, Y)
    if return_X
        return conditional_complete_information_iid(theta, Y, all_Xs), all_Xs
    else
        return conditional_complete_information_iid(theta, Y, all_Xs)
    end
end


"""
Empirical covariance matrix of complete data score.
MC sample provided as argument.
"""
function conditional_complete_score_cov_iid(theta, Y, all_Xs)
    all_scores = [complete_data_score(theta, Y, X) for X in all_Xs]
    return cov(all_scores)
end


"""
Empirical conditional expectation of outer product of complete data score with itself.
MC sample computed internally.
"""
function conditional_complete_score_cov_iid(theta, Y, M, return_X)
    all_Xs = sample_X_given_Y_iid(M, theta, Y)
    if return_X
        return conditional_complete_score_cov_iid(theta, Y, all_Xs), all_Xs
    else
        return conditional_complete_score_cov_iid(theta, Y, all_Xs)
    end
end    


"""
Estimated observed data information matrix from MCEM.
MC sample provided as argument.
"""
function obs_information_formula_iid(theta, Y, all_Xs)
    A = conditional_complete_information_iid(theta, Y, all_Xs)
    B = conditional_complete_score_cov_iid(theta, Y, all_Xs)

    return A - B
end


"""
Estimated observed data information matrix from MCEM.
MC sample computed internally.
"""
function obs_information_formula_iid(theta, Y, M, return_X)
    all_Xs = sample_X_given_Y_iid(M, theta, Y)
    if return_X
        return obs_information_formula_iid(theta, Y, all_Xs), all_Xs
    else
        return obs_information_formula_iid(theta, Y, all_Xs)
    end
end


function MCEM_cov_formula_iid(theta, Y, all_Xs)
    I = obs_information_formula_iid(theta, Y, all_Xs)
    return inv(I)
end

function MCEM_cov_formula_iid(theta, Y, M, return_X)
    all_Xs = sample_X_given_Y_iid(M, theta, Y)
    if return_X
        return MCEM_cov_formula_iid(theta, Y, all_Xs), all_Xs
    else
        return MCEM_cov_formula_iid(theta, Y, all_Xs)
    end
end




# ---------------------------------------------------------------------------- #
#                              Objective Function                              #
# ---------------------------------------------------------------------------- #

"""
Evaluate the MCEM objective function under iid sampling.
"""
function Q_MCEM_iid(theta, Y, all_Xs)
    sum_loglik = 0
    for X in all_Xs
        this_loglik = complete_data_log_lik(theta, Y, X)
        sum_loglik += this_loglik
    end

    return sum_loglik / length(all_Xs)
end


"""
Evaluate the improvement in the MCEM objective function under iid sampling.
"""
function Q_MCEM_increment_iid(theta_new, theta_old, Y, all_Xs)
    A = Q_MCEM_iid(theta_new, Y, all_Xs)
    B = Q_MCEM_iid(theta_old, Y, all_Xs)

    return A - B
end


"""
Evaluate the gradient of the MCEM objective function.
"""
function grad_MCEM_iid(theta, Y, all_Xs)
    sum_score = zeros(2)

    for X in all_Xs
        this_score = complete_data_score(theta, Y, X)
        sum_score += this_score
    end

    return sum_score / length(all_Xs)
end



# ---------------------------------------------------------------------------- #
#                   Run MCEM with fixed number of iterations                   #
# ---------------------------------------------------------------------------- #

function run_MCEM_fixed_iteration_count(theta_init, Y, M, K; return_trajectory=false)
    all_theta_hats = Vector{Vector{Float64}}(undef, K+1)
    all_theta_hats[1] = theta_init

    theta_hat = theta_init

    for i in 1:K
        theta_hat = MCEM_update_iid(theta_hat, Y, M)
        all_theta_hats[i+1] = theta_hat
    end

    if return_trajectory
        return theta_hat, all_theta_hats
    else
        return theta_hat
    end
end


# # ---------------------------------------------------------------------------- #
# #                              Conditional Sampler                             #
# # ---------------------------------------------------------------------------- #

# # ---------------------------- Importance Weights ---------------------------- #

# # Procedure for this section:
# # 1. Compute the importance ratio for a single proposed component of x.
# # 2. Compute the importance ratio for a single proposed X. Return this value on log-scale.
# # 3. Compute the log-importance ratio for all proposed Xs.
# # 4. Compute the log-normalizing constant via the log-sum-exp function.
# # 5. Compute the normalized weights by taking differences and exponentiating. I.e. w = exp(log(w) - log(norm_const))

# # Proposal distribution is same as target distribution, but with variance doubled

# """
# Compute the contribution of one x to its sample's raw importance weight.
# """
# function one_raw_imp_weight_term(x, theta, y, theta_fixed)
#     mu = mu_X_given_Y(theta, y, theta_fixed)
#     var = var_X_given_Y(theta, theta_fixed)

#     f1 = exp(-(x^2 - 2 * mu * x)/(2 * var))
#     f2 = exp(-(x^2 - 2 * mu * x)/(4 * var))

#     return f1 / f2
# end

# """
# Compute the log of this sample's raw importance weight.
# Note: log is used so that normalized importance wights can be calculated via log-sum-exp.
# """
# function one_log_raw_imp_weight(X, theta, Y, theta_fixed)
#     all_raw_imp_weight_terms = [one_raw_imp_weight_term(x, theta, y, theta_fixed) for (x, y) in zip(X, Y)]
#     all_log_raw_imp_weight_terms = log.(all_raw_imp_weight_terms)
#     return sum(all_log_raw_imp_weight_terms)
# end


# """
# Compute all samples' log raw importance weights.
# """
# function get_all_log_raw_imp_weights(all_Xs, theta, Y, theta_fixed)
#     return [one_log_raw_imp_weight(X, theta, Y, theta_fixed) for X in all_Xs]
# end

# # """
# # Normalize and exponentiate the given log weights.
# # """
# # function normalize_weights(all_log_weights)
# #     log_norm_const = logsumexp(all_log_weights)
# #     return exp.(all_log_weights .- log_norm_const)
# # end

# """
# Truncate, then normalize and exponentiate the given log weights.
# """
# function normalize_weights(all_log_weights)
#     log_norm_const = logsumexp(all_log_weights)

#     M = length(all_log_weights)
#     tau = log_norm_const - log(M)/2

#     # all_log_trunc_weights = all_log_weights                             # Don't truncate weights
#     all_log_trunc_weights = [min(w, tau) for w in all_log_weights]    # Truncate weights

#     log_trunc_norm_const = logsumexp(all_log_trunc_weights)

#     return exp.(all_log_trunc_weights .- log_trunc_norm_const)
# end


# """
# Compute the normalized importance weights (not on log-scale).
# """
# function get_all_imp_weights(all_Xs, theta, Y, theta_fixed)
#     all_log_weights = get_all_log_raw_imp_weights(all_Xs, theta, Y, theta_fixed)
#     return normalize_weights(all_log_weights)
# end






# # --------------------- Sample from proposal distribution -------------------- #
# # I.e. Normal with mean zero and variance equal to the sum of the conditional variance and squared conditional mean of X given Y.

# """
# Generate one observation of X (of the same length as Y) from the proposal distribution. 
# """
# function propose_one_X(theta, Y, theta_fixed)
#     all_mus = [mu_X_given_Y(theta, y, theta_fixed) for y in Y]
#     var = var_X_given_Y(theta, theta_fixed)

    
#     this_samp = rand.(Normal.(all_mus, sqrt(2 * var)))

#     # Sample from the multivariate normal distribution with given covariance matrix.
#     return this_samp
# end


# """
# Generate M samples of X (each of the same length as Y) from the proposal distribution. 
# """
# function propose_X(theta, Y, theta_fixed, M)
#     all_Xs = [propose_one_X(theta, Y, theta_fixed) for _ in 1:M]
#     return all_Xs
# end


# """
# Generate an importance sample for X, as well as the corresponding importance weights.
# Note: If raw_weights is true, then the weights are returned unnormalized and on log-scale.
# """
# function get_importance_sample(theta, Y, theta_fixed, M; raw_weights=false)
#     all_Xs = propose_X(theta, Y, theta_fixed, M)
    
#     if !raw_weights
#         all_weights = get_all_imp_weights(all_Xs, theta, Y, theta_fixed)
#     else
#         all_weights = get_all_log_raw_imp_weights(all_Xs, theta, Y, theta_fixed)
#     end

#     # Return weighted sample
#     return all_Xs, all_weights
# end




# # ---------------------------------------------------------------------------- #
# #                             MCEM Update Functions                            #
# # ---------------------------------------------------------------------------- #

# """
# Compute an estimate of sigma^2 using the given estimate of beta and the data.
# """
# function estimate_sigma2(beta, Y, X)
#     all_y_hats = [beta * x for x in X]
#     all_y_diffs = [y - y_hat for (y, y_hat) in zip(Y, all_y_hats)]

#     return sum(all_y_diffs .^ 2) / length(Y)
# end

# """
# Compute the next estimate of theta using MCEM.
# MC sample is provided as an argument.
# """
# function MCEM_update(Y, all_Xs, all_weights)
#     # --------------------------------- Beta hat --------------------------------- #
#     all_cross_prods = [dot(Y, X) for X in all_Xs]
#     all_self_prods = [dot(X, X) for X in all_Xs]

#     beta_hat = dot(all_cross_prods, all_weights) / dot(all_self_prods, all_weights)


#     # --------------------------------- Sigma hat -------------------------------- #
#     all_sigma2_hats = [estimate_sigma2(beta_hat, Y, X) for X in all_Xs]
#     sigma_hat = sqrt(dot(all_sigma2_hats, all_weights))


#     # ------------------------------- Return output ------------------------------ #
#     theta_hat = [beta_hat, sigma_hat]
#     return theta_hat
# end

# """
# Compute the next estimate of theta using MCEM.
# MC sample is generated internally and optionally returned.
# """
# function MCEM_update(theta_old, Y, theta_fixed, M; return_X=false)
#     all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)
#     theta_hat = MCEM_update(Y, all_Xs, all_weights)

#     if !return_X
#         return theta_hat
#     else
#         return theta_hat, all_Xs, all_weights
#     end
# end


# """
# Compute the next estimate of theta using MCEM with iid sampling from the conditional distribution of X.
# MC sample is generated internally and optionally returned.
# """
# function iid_MCEM_update(theta_old, Y, theta_fixed, M; return_X=false)
#     all_cond_mus = [mu_X_given_Y(theta_old, y, theta_fixed) for y in Y]
#     cond_var = var_X_given_Y(theta_old, theta_fixed)
#     cond_sd = sqrt(cond_var)

#     all_Xs = [rand.(Normal.(all_cond_mus, cond_sd)) for _ in 1:M]
#     all_weights = repeat([1/M], M)

#     theta_hat = MCEM_update(Y, all_Xs, all_weights)

#     if !return_X
#         return theta_hat
#     else
#         return theta_hat, all_Xs, all_weights
#     end
# end



# # ---------------------------------------------------------------------------- #
# #                           Objective Function, etc.                           #
# # ---------------------------------------------------------------------------- #


# """
# Objective function of MCEM.
# Supply the a pre-constructed MC sample.
# """
# function Q_MCEM(theta, Y, all_Xs, all_weights, theta_fixed)
#     all_lik = [complete_data_log_lik(theta, Y, X, theta_fixed) for X in all_Xs]

#     return dot(all_lik, all_weights)
# end

# # Q_MCEM(theta1, Y, all_Xs1, all_weights1, theta_fixed)

# """
# Objective function of MCEM.
# MC sample generated inside the function.
# """
# function Q_MCEM(theta, Y, theta_old, theta_fixed, M, return_X)
#     all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)

#     if !return_X
#         return Q_MCEM(theta, Y, all_Xs, all_weights, theta_fixed)
#     else
#         return Q_MCEM(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
#     end
# end

# """
# Compute the difference in log-likelihoods between two values of theta.
# """
# function complete_data_log_lik_increment(theta_new, theta_old, Y, X, theta_fixed)
#     A = complete_data_log_lik(theta_new, Y, X, theta_fixed)
#     B = complete_data_log_lik(theta_old, Y, X, theta_fixed)
#     return A - B
# end

# """
# Improvement in MCEM objective function.
# """
# function Q_MCEM_increment(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed)
#     all_lik_increments = [complete_data_log_lik_increment(theta_new, theta_old, Y, X, theta_fixed) for X in all_Xs]

#     return dot(all_lik_increments, all_weights)
# end






# # ---------------------------------------------------------------------------- #
# #                                Standard error                                #
# # ---------------------------------------------------------------------------- #


# """
# Estimate the conditional expectation of the complete data observed information, given Y=y.
# MC sample provided as argument.
# """
# function MC_complete_cond_info(theta, Y, all_Xs, all_weights, theta_fixed)
#     all_infos = -[complete_data_Hessian(theta, Y, X, theta_fixed) for X in all_Xs]
#     return sum(all_infos .* all_weights)
# end

# """
# Estimate the conditional expectation of the complete data observed information, given Y=y.
# MC sample generated internally.
# """
# function MC_complete_cond_info(theta, Y, theta_old, theta_fixed, M, return_X)
#     all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)
    
#     if !return_X
#         return MC_complete_cond_info(theta, Y, all_Xs, all_weights, theta_fixed)
#     else
#         return MC_complete_cond_info(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
#     end
# end


# """
# Estimate the conditional expectation of the outer product between the complete data score and itself, given Y=y.
# MC sample provided as argument.
# """
# function MC_expect_sq_score(theta, Y, all_Xs, all_weights, theta_fixed)
#     all_scores = [complete_data_score(theta, Y, X, theta_fixed) for X in all_Xs]
#     all_sq_scores = [score * Transpose(score) for score in all_scores]
#     return sum(all_sq_scores .* all_weights)
# end

# """
# Estimate the conditional expectation of the outer product between the complete data score and itself, given Y=y.
# MC sample generated internally.
# """
# function MC_expect_sq_score(theta, Y, theta_old, theta_fixed, M, return_X)
#     all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)
    
#     if !return_X
#         return MC_expect_sq_score(theta, Y, all_Xs, all_weights, theta_fixed)
#     else
#         return MC_expect_sq_score(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
#     end
# end



# """
# Estimate the observed data information using MCEM.
# MC sample provided as argument.
# Note: Formula is only valid at a stationary point of EM.
# """
# function MCEM_obs_data_info_formula(theta, Y, all_Xs, all_weights, theta_fixed)
#     A = MC_complete_cond_info(theta, Y, all_Xs, all_weights, theta_fixed)
#     B = MC_expect_sq_score(theta, Y, all_Xs, all_weights, theta_fixed)

#     output = A - B
#     return output
# end


# """
# Estimate the observed data information using MCEM.
# MC sample generated internally.
# Note: Formula is only valid at a stationary point of EM.
# """
# function MCEM_obs_data_info_formula(theta, Y, theta_old, theta_fixed, M, return_X)
#     all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)
    
#     if !return_X
#         return MCEM_obs_data_info_formula(theta, Y, all_Xs, all_weights, theta_fixed)
#     else
#         return MCEM_obs_data_info_formula(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
#     end
# end


# """
# Estimate the covariance matrix of the MCEM estimate of theta.
# MC sample provided as argument.
# Note: Formula is only valid at a stationary point of EM.
# """
# function MCEM_COV_formula(theta, Y, all_Xs, all_weights, theta_fixed)
#     return inv(MCEM_obs_data_info_formula(theta, Y, all_Xs, all_weights, theta_fixed))
# end

# """
# Estimate the covariance matrix of the MCEM estimate of theta.
# MC sample generated internally.
# Note: Formula is only valid at a stationary point of EM.
# """
# function MCEM_COV_formula(theta, Y, theta_old, theta_fixed, M, return_X)
#     all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)
    
#     if !return_X
#         return MCEM_COV_formula(theta, Y, all_Xs, all_weights, theta_fixed)
#     else
#         return MCEM_COV_formula(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
#     end
# end


# """
# Estimate the standard error of the MCEM estimate of theta. Return a vector of the same length as theta.
# MC sample provided as argument.
# Note: Formula is only valid at a stationary point of EM.
# """
# function MCEM_SE_formula(theta, Y, all_Xs, all_weights, theta_fixed)
#     return sqrt.(diag(MCEM_COV_formula(theta, Y, all_Xs, all_weights, theta_fixed)))
# end


# """
# Estimate the standard error of the MCEM estimate of theta. Return a vector of the same length as theta.
# MC sample generated internally.
# Note: Formula is only valid at a stationary point of EM.
# """
# function MCEM_SE_formula(theta, Y, theta_old, theta_fixed, M, return_X)
#     all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)

#     if !return_X
#         return MCEM_SE_formula(theta, Y, all_Xs, all_weights, theta_fixed)
#     else
#         return MCEM_SE_formula(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
#     end
# end


