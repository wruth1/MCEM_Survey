

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
Complete data Hessian. 
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
#                   Functions for running Booth & Hobert MCEM                  #
# ---------------------------------------------------------------------------- #


function run_MCEM_Booth_Hobert(theta_init, Y, M, K; return_trajectory=false)
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














































# ---------------------------------------------------------------------------- #
#                               Here be dragons.                               #
# ---------------------------------------------------------------------------- #

# From when I was checking that the analytical covariance matrix was correct.

"""
Run the MCEM algorithm B times from the same starting point, theta_old, with M Monte Carlo samples each time.
"""
function many_theta_hats(theta_old, Y, M, B)
    all_theta_hats = []
    prog = Progress(B, desc="Computing theta updates")
    # Threads.@threads for _ in 1:B
    for _ in 1:B
        theta_hat = MCEM_update_iid(theta_old, Y, M; return_X=false)
        push!(all_theta_hats, theta_hat)
        next!(prog)
    end
    return all_theta_hats
end

"""
Empirically determine the mean of theta hat.
"""
function empirical_theta_mean(theta_old, Y, M, B)
    all_theta_hats = []
    prog = Progress(B, desc="Computing theta updates")

    Threads.@threads for _ in 1:B
    # for _ in 1:B
        theta_hat, _ = MCEM_update_iid(theta_old, Y, M; return_X=true)
        push!(all_theta_hats, theta_hat)
        next!(prog)
    end
    return mean(all_theta_hats)
end

"""
Empirically determine the covariance matrix of theta hat
"""
function empirical_theta_cov_mat(theta_old, Y, M, B)
    all_theta_hats = []
    prog = Progress(B, desc="Computing theta updates")

    Threads.@threads for _ in 1:B
    # for _ in 1:B
        theta_hat, _ = MCEM_update_iid(theta_old, Y, M; return_X=true)
        push!(all_theta_hats, theta_hat)
        next!(prog)
    end
    return cov(all_theta_hats)
end



# ---------------------------------------------------------------------------- #
#                                Run Comparison                                #
# ---------------------------------------------------------------------------- #

M = 10000 # Size of each MC sample


# --------------------- First, the analytical covariance --------------------- #
Random.seed!(1)
theta_hat, all_Xs = MCEM_update_iid(theta_init, Y, M; return_X=true)
anal_cov = theta_asymp_cov(theta_hat, Y, all_Xs)



# --------------------- Next, the empirical covariance ---------------------- #

B = 1000 # Number of samples to draw
Random.seed!(1)

#* I tried generating the MC sample once and passing it along all the functions, but doing this with a large enough sample that the simulation isn't just trivial requires more memory than my computer can handle.

# ----------------------------- Compute EM update ---------------------------- #
theta_hat_EM = EM_update(theta_init, Y);

# --------------------- Generate a sample of MCEM updates -------------------- #
all_theta_hats = many_theta_hats(theta_init, Y, M, B);

emp_cov = cov(all_theta_hats)





# ---------------------- A more exact analytical formula --------------------- #

theta_hat_EM = EM_update(theta_init, Y)

# Conditional expectation of complete data Hessian
# Expectation is wrt theta_init, Hessian is evaluated at theta_hat_EM
cond_mu = mu_X_given_Y(theta_init, Y)
cond_H = complete_data_Hessian(theta_hat_EM, Y, cond_mu)

# Conditional expectation of "squared" complete data score
cond_score = expected_squared_score(theta_hat_EM, Y)

# Compute the asymptotic covariance
exact_anal_cov = cond_H^-1 * cond_score * cond_H^-1 / M
