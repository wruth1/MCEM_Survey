

export one_raw_imp_weight_term, one_log_raw_imp_weight, get_all_log_raw_imp_weights, normalize_weights, get_all_imp_weights
export propose_one_X, propose_X, get_importance_sample
export sample_one_X_given_Y, sample_X_given_Y
export estimate_sigma2, MCEM_update
export complete_data_log_lik_increment, Q_MCEM, Q_MCEM_with_sample, Q_MCEM_increment
export MC_complete_cond_info, MC_expect_sq_score,  MCEM_obs_data_info_formula
export MCEM_COV_formula, MCEM_SE_formula



# ---------------------------------------------------------------------------- #
#                              Conditional Sampler                             #
# ---------------------------------------------------------------------------- #

# ---------------------------- Importance Weights ---------------------------- #

# Procedure for this section:
# 1. Compute the importance ratio for a single proposed component of x.
# 2. Compute the importance ratio for a single proposed X. Return this value on log-scale.
# 3. Compute the log-importance ratio for all proposed Xs.
# 4. Compute the log-normalizing constant via the log-sum-exp function.
# 5. Compute the normalized weights by taking differences and exponentiating. I.e. w = exp(log(w) - log(norm_const))

# Proposal distribution is same as target distribution, but with variance doubled

"""
Compute the contribution of one x to its sample's raw importance weight.
"""
function one_raw_imp_weight_term(x, theta, y, theta_fixed)
    mu = mu_X_given_Y(theta, y, theta_fixed)
    var = var_X_given_Y(theta, theta_fixed)

    f1 = exp(-(x^2 - 2 * mu * x)/(2 * var))
    f2 = exp(-(x^2 - 2 * mu * x)/(4 * var))

    return f1 / f2
end

"""
Compute the log of this sample's raw importance weight.
Note: log is used so that normalized importance wights can be calculated via log-sum-exp.
"""
function one_log_raw_imp_weight(X, theta, Y, theta_fixed)
    all_raw_imp_weight_terms = [one_raw_imp_weight_term(x, theta, y, theta_fixed) for (x, y) in zip(X, Y)]
    all_log_raw_imp_weight_terms = log.(all_raw_imp_weight_terms)
    return sum(all_log_raw_imp_weight_terms)
end


"""
Compute all samples' log raw importance weights.
"""
function get_all_log_raw_imp_weights(all_Xs, theta, Y, theta_fixed)
    return [one_log_raw_imp_weight(X, theta, Y, theta_fixed) for X in all_Xs]
end

# """
# Normalize and exponentiate the given log weights.
# """
# function normalize_weights(all_log_weights)
#     log_norm_const = logsumexp(all_log_weights)
#     return exp.(all_log_weights .- log_norm_const)
# end

"""
Truncate, then normalize and exponentiate the given log weights.
"""
function normalize_weights(all_log_weights)
    log_norm_const = logsumexp(all_log_weights)

    M = length(all_log_weights)
    tau = log_norm_const - log(M)/2

    all_log_trunc_weights = [min(w, tau) for w in all_log_weights]
    log_trunc_norm_const = logsumexp(all_log_trunc_weights)

    return exp.(all_log_trunc_weights .- log_trunc_norm_const)
end


"""
Compute the normalized importance weights (not on log-scale).
"""
function get_all_imp_weights(all_Xs, theta, Y, theta_fixed)
    all_log_weights = get_all_log_raw_imp_weights(all_Xs, theta, Y, theta_fixed)
    return normalize_weights(all_log_weights)
end






# --------------------- Sample from proposal distribution -------------------- #
# I.e. Normal with mean zero and variance equal to the sum of the conditional variance and squared conditional mean of X given Y.

"""
Generate one observation of X (of the same length as Y) from the proposal distribution. 
"""
function propose_one_X(theta, Y, theta_fixed)
    all_mus = [mu_X_given_Y(theta, y, theta_fixed) for y in Y]
    var = var_X_given_Y(theta, theta_fixed)

    
    this_samp = rand.(Normal.(all_mus, sqrt(2 * var)))

    # Sample from the multivariate normal distribution with given covariance matrix.
    return this_samp
end


"""
Generate M samples of X (each of the same length as Y) from the proposal distribution. 
"""
function propose_X(theta, Y, theta_fixed, M)
    all_Xs = [propose_one_X(theta, Y, theta_fixed) for _ in 1:M]
    return all_Xs
end


"""
Generate an importance sample for X, as well as the corresponding importance weights.
Note: If raw_weights is true, then the weights are returned unnormalized and on log-scale.
"""
function get_importance_sample(theta, Y, theta_fixed, M; raw_weights=false)
    all_Xs = propose_X(theta, Y, theta_fixed, M)
    
    if !raw_weights
        all_weights = get_all_imp_weights(all_Xs, theta, Y, theta_fixed)
    else
        all_weights = get_all_log_raw_imp_weights(all_Xs, theta, Y, theta_fixed)
    end

    # Return weighted sample
    return all_Xs, all_weights
end

#! Start Here
# todo: Replace sample_X_given_Y with get_importance_sample throughout the project. This will require updating any functions which use this sample to incorporate weights as well.


# ------------------ For comparison, include direct samplers ----------------- #

"""
Generate a single sample directly from the conditional distribution of X given Y (of length equal to length(Y)). I.e. No proposal distribution.
"""
function sample_one_X_given_Y(theta, Y, theta_fixed)
    all_means = [mu_X_given_Y(theta, y, theta_fixed) for y in Y]
    var = var_X_given_Y(theta, theta_fixed)

    this_X = rand.(Normal.(all_means, sqrt(var)))
    return this_X
end


"""
Sample directly from the conditional distribution of X given Y (of length equal to length(Y)). I.e. No proposal distribution.
"""
function sample_X_given_Y(theta, Y, theta_fixed, n)
    output = [sample_one_X_given_Y(theta, Y, theta_fixed) for i in 1:n]
    return output
end



# ---------------------------------------------------------------------------- #
#                             MCEM Update Functions                            #
# ---------------------------------------------------------------------------- #

"""
Compute an estimate of sigma^2 using the given estimate of beta and the data.
"""
function estimate_sigma2(beta, Y, X)
    all_y_hats = [beta * x for x in X]
    all_y_diffs = [y - y_hat for (y, y_hat) in zip(Y, all_y_hats)]

    return sum(all_y_diffs .^ 2) / length(Y)
end

"""
Compute the next estimate of theta using MCEM.
MC sample is provided as an argument.
"""
function MCEM_update(Y, all_Xs, all_weights)
    # --------------------------------- Beta hat --------------------------------- #
    all_cross_prods = [dot(Y, X) for X in all_Xs]
    all_self_prods = [dot(X, X) for X in all_Xs]

    beta_hat = dot(all_cross_prods, all_weights) / dot(all_self_prods, all_weights)


    # --------------------------------- Sigma hat -------------------------------- #
    all_sigma2_hats = [estimate_sigma2(beta_hat, Y, X) for X in all_Xs]
    sigma_hat = sqrt(dot(all_sigma2_hats, all_weights))


    # ------------------------------- Return output ------------------------------ #
    theta_hat = [beta_hat, sigma_hat]
    return theta_hat
end

"""
Compute the next estimate of theta using MCEM.
MC sample is generated internally and optionally returned.
"""
function MCEM_update(theta_old, Y, theta_fixed, M; return_X=false)
    all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)
    theta_hat = MCEM_update(Y, all_Xs, all_weights)

    if !return_X
        return theta_hat
    else
        return theta_hat, all_Xs, all_weights
    end
end




# ---------------------------------------------------------------------------- #
#                           Objective Function, etc.                           #
# ---------------------------------------------------------------------------- #


"""
Objective function of MCEM.
Supply the a pre-constructed MC sample.
"""
function Q_MCEM(theta, Y, all_Xs, all_weights, theta_fixed)
    all_lik = [complete_data_log_lik(theta, Y, X, theta_fixed) for X in all_Xs]

    return dot(all_lik, all_weights)
end

# Q_MCEM(theta1, Y, all_Xs1, all_weights1, theta_fixed)

"""
Objective function of MCEM.
MC sample generated inside the function.
"""
function Q_MCEM(theta, Y, theta_old, theta_fixed, M, return_X)
    all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)

    if !return_X
        return Q_MCEM(theta, Y, all_Xs, all_weights, theta_fixed)
    else
        return Q_MCEM(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
    end
end

"""
Compute the difference in log-likelihoods between two values of theta.
"""
function complete_data_log_lik_increment(theta_new, theta_old, Y, X, theta_fixed)
    A = complete_data_log_lik(theta_new, Y, X, theta_fixed)
    B = complete_data_log_lik(theta_old, Y, X, theta_fixed)
    return A - B
end

"""
Improvement in MCEM objective function.
"""
function Q_MCEM_increment(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed)
    all_lik_increments = [complete_data_log_lik_increment(theta_new, theta_old, Y, X, theta_fixed) for X in all_Xs]

    return dot(all_lik_increments, all_weights)
end






# ---------------------------------------------------------------------------- #
#                                Standard error                                #
# ---------------------------------------------------------------------------- #


"""
Estimate the conditional expectation of the complete data observed information, given Y=y.
MC sample provided as argument.
"""
function MC_complete_cond_info(theta, Y, all_Xs, all_weights, theta_fixed)
    all_infos = -[complete_data_Hessian(theta, Y, X, theta_fixed) for X in all_Xs]
    return sum(all_infos .* all_weights)
end

"""
Estimate the conditional expectation of the complete data observed information, given Y=y.
MC sample generated internally.
"""
function MC_complete_cond_info(theta, Y, theta_old, theta_fixed, M, return_X)
    all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)
    
    if !return_X
        return MC_complete_cond_info(theta, Y, all_Xs, all_weights, theta_fixed)
    else
        return MC_complete_cond_info(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
    end
end


"""
Estimate the conditional expectation of the outer product between the complete data score and itself, given Y=y.
MC sample provided as argument.
"""
function MC_expect_sq_score(theta, Y, all_Xs, all_weights, theta_fixed)
    all_scores = [complete_data_score(theta, Y, X, theta_fixed) for X in all_Xs]
    all_sq_scores = [score * Transpose(score) for score in all_scores]
    return sum(all_sq_scores .* all_weights)
end

"""
Estimate the conditional expectation of the outer product between the complete data score and itself, given Y=y.
MC sample generated internally.
"""
function MC_expect_sq_score(theta, Y, theta_old, theta_fixed, M, return_X)
    all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)
    
    if !return_X
        return MC_expect_sq_score(theta, Y, all_Xs, all_weights, theta_fixed)
    else
        return MC_expect_sq_score(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
    end
end



"""
Estimate the observed data information using MCEM.
MC sample provided as argument.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_obs_data_info_formula(theta, Y, all_Xs, all_weights, theta_fixed)
    A = MC_complete_cond_info(theta, Y, all_Xs, all_weights, theta_fixed)
    B = MC_expect_sq_score(theta, Y, all_Xs, all_weights, theta_fixed)

    output = A - B
    return output
end


"""
Estimate the observed data information using MCEM.
MC sample generated internally.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_obs_data_info_formula(theta, Y, theta_old, theta_fixed, M, return_X)
    all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)
    
    if !return_X
        return MCEM_obs_data_info_formula(theta, Y, all_Xs, all_weights, theta_fixed)
    else
        return MCEM_obs_data_info_formula(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
    end
end


"""
Estimate the covariance matrix of the MCEM estimate of theta.
MC sample provided as argument.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_COV_formula(theta, Y, all_Xs, all_weights, theta_fixed)
    return inv(MCEM_obs_data_info_formula(theta, Y, all_Xs, all_weights, theta_fixed))
end

"""
Estimate the covariance matrix of the MCEM estimate of theta.
MC sample generated internally.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_COV_formula(theta, Y, theta_old, theta_fixed, M, return_X)
    all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)
    
    if !return_X
        return MCEM_COV_formula(theta, Y, all_Xs, all_weights, theta_fixed)
    else
        return MCEM_COV_formula(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
    end
end


"""
Estimate the standard error of the MCEM estimate of theta. Return a vector of the same length as theta.
MC sample provided as argument.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_SE_formula(theta, Y, all_Xs, all_weights, theta_fixed)
    return sqrt.(diag(MCEM_COV_formula(theta, Y, all_Xs, all_weights, theta_fixed)))
end


"""
Estimate the standard error of the MCEM estimate of theta. Return a vector of the same length as theta.
MC sample generated internally.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_SE_formula(theta, Y, theta_old, theta_fixed, M, return_X)
    all_Xs, all_weights = get_importance_sample(theta_old, Y, theta_fixed, M)

    if !return_X
        return MCEM_SE_formula(theta, Y, all_Xs, all_weights, theta_fixed)
    else
        return MCEM_SE_formula(theta, Y, all_Xs, all_weights, theta_fixed), all_Xs, all_weights
    end
end


