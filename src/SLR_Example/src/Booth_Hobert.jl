
export theta_ASE_middle_term, theta_ASE_outer_term, theta_asymp_cov
export get_chisq_quantile, get_Mahal_dist, check_sufficient_MC
export stop_check
export run_MCEM_Booth_Hobert


# ---------------------------------------------------------------------------- #
#                         Asymptotic SE of MCEM update                         #
# ---------------------------------------------------------------------------- #

"""
Expected squared complete data score (i.e. outer product with itself)
"""
function theta_ASE_middle_term(theta_new, Y, all_Xs)
    all_scores = [complete_data_score(theta_new, Y, X) for X in all_Xs]
    all_score_prods = [score * score' for score in all_scores]    # Outer product of each score vector with itself
    mean_score_prod = mean(all_score_prods)
    return mean_score_prod
end

"""
Inverse of complete data Hessian. 
"""
function theta_ASE_outer_term(theta_new, Y, all_Xs)
    all_hessians = [complete_data_Hessian(theta_new, Y, X) for X in all_Xs]
    mean_hessian = mean(all_hessians)

    return mean_hessian^-1
end


"""
Estimate the asymptotic covariance of the MCEM update around the EM update
"""
function theta_asymp_cov(theta_new, Y, all_Xs)
    A = theta_ASE_middle_term(theta_new, Y, all_Xs)
    B = theta_ASE_outer_term(theta_new, Y, all_Xs)

    M = length(all_Xs)

    # Apply Hermitian() to the output to fix some roundoff error
    return Hermitian(B * A * B ./ M)
end


# ---------------------------------------------------------------------------- #
#                Check whether the MC size is sufficiently large               #
# ---------------------------------------------------------------------------- #

# The idea is to use the MCEM update to construct a confidence ellipsoid for what an EM update would have been. If this ellipsoid contains the previous iteration's estimate, then there is too much variability and we need to increase the MC size.

"""
Get the chi-squared quantile for a given alpha and degrees of freedom
Used to construct confidence ellipsoids for the standard multivariate normal
"""
function get_chisq_quantile(alpha, df)
    return quantile(Chisq(df), 1 - alpha)
end

"""
Compute the squared Mahalanobis distance corresponding to an MCEM update.
"""
function get_Mahal_dist(theta_new, theta_old, Sigma_hat)
    theta_diff = theta_new - theta_old
    return theta_diff' * Sigma_hat^-1 * theta_diff
end


"""
Check whether the previous iteration's parameter estimate is inside the confidence ellipsoid from this iteration's estimate.
"""
function check_sufficient_MC(theta_new, theta_old, Y, all_Xs, alpha)
    # Compute squared Mahalanobis distance between new and old estimates
    asymp_cov = theta_asymp_cov(theta_new, Y, all_Xs)
    update_size = get_Mahal_dist(theta_new, theta_old, asymp_cov)

    # Compute chi-squared quantile
    chisq_quantile = get_chisq_quantile(alpha, length(theta_new))

    # Check if update is outside confidence ellipsoid
    # Note: this corresponds to a large update from MCEM, so we are happy with our MC size.
    return update_size > chisq_quantile
end

# ---------------------------------------------------------------------------- #
#                   Functions for running Booth & Hobert MCEM                  #
# ---------------------------------------------------------------------------- #


"""
Checks whether relative error condition from paper is satisfied.
"""
function stop_check(theta_new, theta_old, delta, tau)
    rel_errs = abs.(theta_new - theta_old) ./ (theta_old .+ delta)
    return maximum(rel_errs) < tau
end


"""
Run the MCEM algorithm from Booth & Hobert (1999) until its termination criterion is satisfied.
"""
function run_MCEM_Booth_Hobert(theta_init, Y; M_init = 10, alpha = 0.25, k = 3, tau = 0.002, delta = 0.001, return_trajectory=false, return_diagnostics = false)
    M = M_init
    
    all_theta_hats = []
    all_Ms = []

    theta_hat_old = theta_init 

    stop_check_count = 0    # Number of consecutive iterations with sufficiently small updates
    iteration_count = 0

    while stop_check_count < 3
        iteration_count += 1
        # println("Iteration $iteration_count")

        push!(all_Ms, M)

        theta_hat, all_Xs = MCEM_update_iid(theta_hat_old, Y, M; return_X=true)
        push!(all_theta_hats, theta_hat)

        # Check whether MC size is sufficient
        this_sufficient_MC = check_sufficient_MC(theta_hat, theta_hat_old, Y, all_Xs, alpha)
        if !this_sufficient_MC
            M += ceil(M / k)
        end

        # Check whether we should stop iterating. I.e. Whether the relative size of the update is small enough.
        this_stop_check = stop_check(theta_hat, theta_hat_old, delta, tau)
        if this_stop_check
            stop_check_count += 1
        else
            theta_hat_old = theta_hat
            stop_check_count = 0
        end
    end

    if return_trajectory && return_diagnostics
        return all_theta_hats, iteration_count, all_Ms
    elseif return_trajectory && !return_diagnostics
        return all_theta_hats
    elseif !return_trajectory && return_diagnostics
        return theta_hat, iteration_count, all_Ms
    else
        return theta_hat
    end
end














































# ---------------------------------------------------------------------------- #
#                               Here be dragons.                               #
# ---------------------------------------------------------------------------- #

# # From when I was checking that the analytical covariance matrix was correct.

# """
# Run the MCEM algorithm B times from the same starting point, theta_old, with M Monte Carlo samples each time.
# """
# function many_theta_hats(theta_old, Y, M, B)
#     all_theta_hats = []
#     prog = Progress(B, desc="Computing theta updates")
#     # Threads.@threads for _ in 1:B
#     for _ in 1:B
#         theta_hat = MCEM_update_iid(theta_old, Y, M; return_X=false)
#         push!(all_theta_hats, theta_hat)
#         next!(prog)
#     end
#     return all_theta_hats
# end

# """
# Empirically determine the mean of theta hat.
# """
# function empirical_theta_mean(theta_old, Y, M, B)
#     all_theta_hats = []
#     prog = Progress(B, desc="Computing theta updates")

#     Threads.@threads for _ in 1:B
#     # for _ in 1:B
#         theta_hat, _ = MCEM_update_iid(theta_old, Y, M; return_X=true)
#         push!(all_theta_hats, theta_hat)
#         next!(prog)
#     end
#     return mean(all_theta_hats)
# end

# """
# Empirically determine the covariance matrix of theta hat
# """
# function empirical_theta_cov_mat(theta_old, Y, M, B)
#     all_theta_hats = []
#     prog = Progress(B, desc="Computing theta updates")

#     Threads.@threads for _ in 1:B
#     # for _ in 1:B
#         theta_hat, _ = MCEM_update_iid(theta_old, Y, M; return_X=true)
#         push!(all_theta_hats, theta_hat)
#         next!(prog)
#     end
#     return cov(all_theta_hats)
# end



# # ---------------------------------------------------------------------------- #
# #                                Run Comparison                                #
# # ---------------------------------------------------------------------------- #

# M = 10000 # Size of each MC sample


# # --------------------- First, the analytical covariance --------------------- #
# Random.seed!(1)
# theta_hat, all_Xs = MCEM_update_iid(theta_init, Y, M; return_X=true)
# anal_cov = theta_asymp_cov(theta_hat, Y, all_Xs)



# # --------------------- Next, the empirical covariance ---------------------- #

# B = 1000 # Number of samples to draw
# Random.seed!(1)

# #* I tried generating the MC sample once and passing it along all the functions, but doing this with a large enough sample that the simulation isn't just trivial requires more memory than my computer can handle.

# # ----------------------------- Compute EM update ---------------------------- #
# theta_hat_EM = EM_update(theta_init, Y);

# # --------------------- Generate a sample of MCEM updates -------------------- #
# all_theta_hats = many_theta_hats(theta_init, Y, M, B);

# emp_cov = cov(all_theta_hats)





# # ---------------------- A more exact analytical formula --------------------- #
# # Somehow, this formula is farther from the empirical covariance than the analytical formula evaluated at the initial value of theta.

# theta_hat_EM = EM_update(theta_init, Y)

# # Conditional expectation of complete data Hessian
# # Expectation is wrt theta_init, Hessian is evaluated at theta_hat_EM
# cond_mu = mu_X_given_Y(theta_init, Y)
# cond_H = complete_data_Hessian(theta_hat_EM, Y, cond_mu)

# # Conditional expectation of "squared" complete data score
# cond_score = expected_squared_score(theta_hat_EM, Y)

# # Compute the asymptotic covariance
# exact_anal_cov = cond_H^-1 * cond_score * cond_H^-1 / M
