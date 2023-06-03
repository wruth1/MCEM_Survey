
export EM_update, run_EM
export expected_complete_info
export get_allele_weights, get_M_score, expected_squared_score, cov_score
export EM_obs_data_information_formula, EM_COV_formula, EM_SE_formula
export Q_EM




# ---------------------------------------------------------------------------- #
#                              EM Update Functions                             #
# ---------------------------------------------------------------------------- #


"""
Perform one iteration of EM.
"""
function EM_update(theta_old, Y)

    # Compute the conditional expectations
    cond_mu = mu_X_given_Y(theta_old, Y)

    # The complete data log-likelihood is linear in X, so we can just replace the components of X with their conditional expectations in the complete data MLE.
    theta_hat = complete_data_MLE(Y, cond_mu)

    return theta_hat
end


"""
Iterate EM from the specified intial value of theta until convergence within the specified relative tolerance.
"""
function run_EM(theta_init, Y; rtol = 1e-6, return_iteration_count = false, return_trajectory = false)
    theta_old = theta_init
    theta_new = EM_update(theta_old, Y)

    if return_trajectory
        trajectory = [theta_old, theta_new]
    end

    count = 1

    while (norm(theta_new - theta_old)/norm(theta_old)) > rtol
        count += 1
        theta_old = theta_new
        theta_new = EM_update(theta_old, Y)

        if return_trajectory
            push!(trajectory, theta_new)
        end
    end


    if return_trajectory && return_iteration_count
        return theta_new, count, trajectory
    elseif return_trajectory && !return_iteration_count
        return theta_new, trajectory
    elseif !return_trajectory && return_iteration_count
        return theta_new, count
    else
        return theta_new
    end
end



# ---------------------------------------------------------------------------- #
#                            Standard error formula                            #
# ---------------------------------------------------------------------------- #




# ------------------------------- Hessian term ------------------------------- #
"""
Conditional expectation of complete data information matrix.
"""
function expected_complete_info(theta, Y)
    cond_mu = mu_X_given_Y(theta, Y)

    H = complete_data_Hessian(theta, Y, cond_mu)
    I = - H

    return I
end


# -------------------------------- Score term -------------------------------- #
"""
Construct vectors of weights which map from X to the number of alleles of each type.
"""
function get_allele_weights()
    eta_O = [2, 1, 0, 1, 0, 0]
    eta_A = [0, 1, 2, 0, 0, 1]
    eta_B = [0, 0, 0, 1, 2, 1]

    return eta_O, eta_A, eta_B
end

"""
Matrix which maps X to the score vector.
"""
function get_M_score(theta)
    eta_O, eta_A, eta_B = get_allele_weights()

    p, q = theta
    r = 1 - p - q

    M1 = eta_A/p - eta_O/r
    M2 = eta_B/q - eta_O/r

    M = [M1 M2]'
    return M
end

"""
Conditional expectation of outer product of score with itself.
"""
function expected_squared_score(theta, Y)
    M = get_M_score(theta)

    cond_sq_X = mu2_X_given_Y(theta, Y)

    cov_score = M * cond_sq_X * M'
    return cov_score
end


"""
Conditional covariance of the score function given Y.
"""
function cov_score(theta, Y)
    M = get_M_score(theta)

    cov_X = cov_X_given_Y(theta, Y)

    cov_score = M * cov_X * M'
    return cov_score
end


# -------------------------- Compute standard error -------------------------- #

"""
Evaluate the formula for the information matrix of the observed data log-likelihood using only quantities available to the EM algorithm.
Note: This formula is only valid at a fixed point of EM.
"""
function EM_obs_data_information_formula(theta, Y)
    A = expected_complete_info(theta, Y)
    B = expected_squared_score(theta, Y)

    return A - B
end



"""
Evaluate the formula for the estimated covariance matrix of the EM estimator. That is, the inverse negative Hessian of the observed data log-likelihood.
Note: This formula is only valid at a fixed point of EM.
"""
function EM_COV_formula(theta, Y)
    return inv(EM_obs_data_information_formula(theta, Y))
end

"""
Evaluate the formula for the estimated standard error of the EM estimator.
Note: This formula is only valid at a fixed point of EM.
"""
function EM_SE_formula(theta, Y)
    return sqrt.(diag(EM_COV_formula(theta, Y)))
end






# ---------------------------------------------------------------------------- #
#                             EM Objective Function                            #
# ---------------------------------------------------------------------------- #

"""
Evaluate the EM objective function at theta, using conditional expectations computed under theta_old.
"""
function Q_EM(theta, Y, theta_old)
    cond_mu = mu_X_given_Y(theta_old, Y)
    return complete_data_log_lik(theta, Y, cond_mu)
end