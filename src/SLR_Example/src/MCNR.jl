
export obs_score_iid
export MCNR_step, MCNR_update
export get_MCNR_W, MCNR_check_convergence
export one_MCNR_iteration, run_MCNR


# ----------------------------- Helper Functions ----------------------------- #

"""
Estimate observed data score function as Monte Carlo average of complete data score function.  Logit parameterization.
"""
function obs_score_iid_logit(theta1, Y, all_Xs)
    score = [0, 0]
    for i in eachindex(all_Xs)
        score += complete_data_score_logit(theta1, Y, all_Xs[i])
    end
    score /= length(all_Xs)
    return score
end



# ----------- Second-derivatives for re-parameterized logit model. ----------- #

"""
Complete data observed information matrix for a single MC sample. Logit parameterization.
"""
function one_complete_data_hessian_logit(theta1, Y, X)
    p, q = logistic.(theta1)

    _, num_A, num_B = num_OAB_alleles(X)

    this_hess = - [num_A * p * (1-p)    0; 
                    0    num_B * q * (1-q)]
    return this_hess
end

"""
Mean of complete data observed information matrix over all MC samples. Logit parameterization.
"""
function complete_data_hessian_logit(theta1, Y, all_Xs)
    mean_hess = [0 0; 0 0]
    for X in all_Xs
        mean_hess += one_complete_data_hessian_logit(theta1, Y, X)
    end
    mean_hess /= length(all_Xs)
    return mean_hess
end

"""
Covariance matrix of complete data score across all MC samples. Logit parameterization.
"""
function complete_score_cov_logit(theta1, Y, all_Xs)
    all_scores_logit = [complete_data_score_logit(theta1, Y, X) for X in all_Xs]
    score_cov_logit = cov(all_scores_logit)
    return score_cov_logit
end

"""
Estimated observed data information matrix. Logit parameterization.
"""
function obs_information_iid_logit(theta1, Y, all_Xs)
    mean_hess_logit = complete_data_hessian_logit(theta1, Y, all_Xs)
    score_cov_logit = complete_score_cov_logit(theta1, Y, all_Xs)
    return mean_hess_logit - score_cov_logit
end


# ------------------------------- Update theta1 ------------------------------- #

"""
Compute the step (direction and length) for Monte Carlo Newton-Raphson.
"""
function MCNR_step(theta1, Y, all_Xs)
    this_grad = obs_score_iid_logit(theta1, Y, all_Xs)
    this_info = obs_information_iid_logit(theta1, Y, all_Xs)
    return this_info \ this_grad
end

"""
Update theta1 using Monte Carlo Newton-Raphson.
"""
function MCNR_update(theta1, Y, all_Xs)
    step = MCNR_step(theta1, Y, all_Xs)
    return theta1 - step
end



# ------------------------------- Stopping Rule ------------------------------ #

"""
Return test statistic for obs data score being non-zero.
"""
function get_MCNR_W(theta1, Y, all_Xs)
    all_obs_scores = [complete_data_score_logit(theta1, Y, X) for X in all_Xs]
    score_hat = mean(all_obs_scores)

    score_cov_hat = cov(all_obs_scores)

    this_W = score_hat' * ( score_cov_hat \ score_hat)

    return this_W
end


"""
Assess convergence of MCNR. Tolerance is governed by alpha (smaller values are more strict).
"""
function MCNR_check_convergence(theta1, Y, all_Xs, alpha=0.1)
    this_W = get_MCNR_W(theta1, Y, all_Xs)
    thresh = quantile(Chisq(length(theta1)), 1 - alpha)

    return this_W < thresh
end



# --------------------------------- Run MCNR --------------------------------- #

function one_MCNR_iteration(theta1_old, Y, M; alpha=0.1)
    theta = logistic.(theta1_old)
    println("theta = $theta")
    all_Xs = sample_X_given_Y_iid(M, theta, Y)
    theta1_new = MCNR_update(theta1_old, Y, all_Xs)
    converged = MCNR_check_convergence(theta1_new, Y, all_Xs, alpha)
    return theta1_new, converged
end

function run_MCNR(theta_init, Y, M; alpha=0.1, return_traj=false)
    theta1_old = logit.(theta_init)
    converged = false

    if return_traj
        logit_traj = []
    end

    j=0

    while !converged
        theta1_new, converged = one_MCNR_iteration(theta1_old, Y, M, alpha=alpha)
        theta1_old = theta1_new
        j += 1
        println("Iteration $j")

        if return_traj
            push!(logit_traj, theta1_new)
        end
    end

    traj = logistic.(logit_traj)

    if return_traj
        return traj
    else
        return theta_new
    end
end


