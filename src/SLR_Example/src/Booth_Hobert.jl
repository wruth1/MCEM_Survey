

# ---------------------------------------------------------------------------- #
#                 Asymptotic SE of increment in MCEM objective                 #
# ---------------------------------------------------------------------------- #

M = 10000
theta_hat, all_Xs, all_weights = MCEM_update(theta4, Y, theta_fixed, M; return_X=true)
theta_new = theta_hat

"""
Expected squared complete data score (i.e. outer product with itself)
"""
function theta_ASE_middle_term(theta_new, Y, all_Xs, all_weights, theta_fixed)
    all_scores = [complete_data_score(theta_new, Y, X, theta_fixed) for X in all_Xs]
    all_score_prods = [score * score' for score in all_scores]    # Outer product of each score vector with itself
    mean_score_prod = sum(all_weights .* all_score_prods)
    return mean_score_prod
end

"""
Complete data Hessian. 
"""
function theta_ASE_outer_term(theta_new, Y, all_Xs, all_weights, theta_fixed)
    all_hessians = [complete_data_Hessian(theta_new, Y, X, theta_fixed) for X in all_Xs]
    mean_hessian = sum(all_weights .* all_hessians)

    # At a maximizer, the off-diagonal Hessian entries are zero
    mean_hessian[1,2] = 0.0
    mean_hessian[2,1] = 0.0

    return mean_hessian^-1
end


"""
Estimate the asymptotic covariance of the MCEM update around the EM update
"""
function theta_asymp_cov(theta_new, Y, all_Xs, all_weights, theta_fixed)
    A = theta_ASE_middle_term(theta_new, Y, all_Xs, all_weights, theta_fixed)
    B = theta_ASE_outer_term(theta_new, Y, all_Xs, all_weights, theta_fixed)

    # Apply Hermitian() to the output to fix roundoff error
    return Hermitian(B * A * B)
end


"""
Estimate the covariance matrix of theta hat (adjusting for sample size)
"""
function theta_cov_mat(theta_new, Y, all_Xs, all_weights, theta_fixed)
    Sigma_asymp = theta_asymp_cov(theta_new, Y, all_Xs, all_weights, theta_fixed)
    Sigma = Sigma_asymp / get_ESS(all_weights)
    return Sigma
end

"""
Run the MCEM algorithm B times from the same starting point, theta_old, with M Monte Carlo samples each time.
"""
function many_theta_hats(theta_old, Y, M, B, theta_fixed)
    all_theta_hats = []
    all_ESSs = []
    prog = Progress(B, desc="Computing theta updates")
    # Threads.@threads for _ in 1:B
    for _ in 1:B
        theta_hat, _, this_weights = MCEM_update(theta_old, Y, theta_fixed, M; return_X=true)
        this_ESS = get_ESS(this_weights)
        push!(all_theta_hats, theta_hat)
        push!(all_ESSs, this_ESS)
        next!(prog)
    end
    return [all_theta_hats, all_ESSs]
end

"""
Empirically determine the mean of theta hat.
"""
function empirical_theta_mean(theta_old, Y, M, B, theta_fixed)
    all_theta_hats = []
    all_ESSs = []
    prog = Progress(B, desc="Computing theta updates")
    Threads.@threads for _ in 1:B
    # for _ in 1:B
        theta_hat, _, this_weights = MCEM_update(theta_old, Y, theta_fixed, M; return_X=true)
        this_ESS = get_ESS(this_weights)
        push!(all_theta_hats, theta_hat)
        push!(all_ESSs, this_ESS)
        next!(prog)
    end
    return [mean(all_theta_hats), all_ESSs]
end

"""
Empirically determine the covariance matrix of theta hat
"""
function empirical_theta_cov_mat(theta_old, Y, M, B, theta_fixed)
    all_theta_hats = []
    all_ESSs = []
    prog = Progress(B, desc="Computing theta updates")
    Threads.@threads for _ in 1:B
    # for _ in 1:B
        theta_hat, _, this_weights = MCEM_update(theta_old, Y, theta_fixed, M; return_X=true)
        this_ESS = get_ESS(this_weights)
        push!(all_theta_hats, theta_hat)
        push!(all_ESSs, this_ESS)
        next!(prog)
    end
    return [cov(all_theta_hats), all_ESSs]
end

###! START HERE
#! The analytical and theoretical standard errors for a single theta hat update are not matching
#! I adjusted the analytical formula to use ESS instead of M. I'm not sure if there's a justification for doing so, but the matrices are much closer this way
#! Furthermore, the iterate from a single MCEM step does not appear to match what we get from the corresponding EM step. I'm not sure if this is a coding problem or a theory problem. The model is pretty nice though, so it's probably a coding problem.

# ----------------------------- Compute EM update ---------------------------- #
theta_hat_EM = EM_update(theta1, Y, theta_fixed);

# --------------------- Generate a sample of MCEM updates -------------------- #
B = 1000 # Number of samples to draw
M = 10000 # Size of each sample
Random.seed!(1)
all_theta_hats, all_ESSs = many_theta_hats(theta1, Y, M, B, theta_fixed);


# ------------- Plot empirical sampling distribution of theta hat ------------ #
# using Plots
all_beta_hats = [theta_hat[1] for theta_hat in all_theta_hats]
all_sigma_hats = [theta_hat[2] for theta_hat in all_theta_hats]

# Marginal histograms
beta_hat_hist = histogram(all_beta_hats, label = nothing, title="beta", bins = 21);
vline!(theta_hat_EM[1:1], label = "EM Update", linewidth = 2);
vline!([mean(all_beta_hats)], label = "Mean MCEM Update", linewidth = 2);

sigma_hat_hist = histogram(all_sigma_hats, label = nothing, title="sigma", bins = 21);
vline!(theta_hat_EM[2:2], label = "EM Update", linewidth = 2);
vline!([mean(all_sigma_hats)], label = "Mean MCEM Update", linewidth = 2);

plot(beta_hat_hist, sigma_hat_hist, layout=(1,2), size=(1200,1000))

# Joint histogram
joint_hist = histogram2d(all_beta_hats, all_sigma_hats, bins = (50, 50), title="Joint distribution of beta and sigma", xlabel="beta", ylabel="sigma", size = (1200, 1000), show_empty_bins=false);
plot!(theta_hat_EM[1:1], theta_hat_EM[2:2], seriestype = :scatter, markersize = 10, label = "EM update");
plot(joint_hist)



# ------------ Check mean of theta hat under sampling distribution ----------- #
B = 100 # Number of samples to draw
M = 100000 # Size of each sample
Random.seed!(1)
emp_mean, all_ESSs = empirical_theta_mean(theta1, Y, M, B, theta_fixed);
emp_mean
anal_mean = EM_update(theta1, Y, theta_fixed)


# --------- Check SE matrix of theta hat under sampling distribution --------- #
q, all_ESSs = empirical_theta_cov_mat(theta4, Y, M, 10, theta_fixed)
q
w = theta_cov_mat(theta_hat, Y, all_Xs, all_weights, theta_fixed)
q2 = sqrt(q)
w2 = sqrt(w)

all_ESSs


"""
Estimate the asymptotic standard error of the increment in MCEM objective function directly from the log-likelihood increments.
"""
function get_resamp_ASE(theta_new, theta_old, Y, all_resamp_Xs, theta_fixed)

    # test_all_Xs = wsample(all_Xs, all_weights, M, replace = true)

    M = length(all_resamp_Xs)
    test_lik_increments = [complete_data_log_lik_increment(theta_new, theta_old, Y, X, theta_fixed) for X in all_resamp_Xs]
    return std(test_lik_increments)/sqrt(M)

end

# ---------------------------------------------------------------------------- #
#                    Investigate ASE formula for IID sample                    #
# ---------------------------------------------------------------------------- #

# Random.seed!(1)

# all_Xs_iid, _ = get_importance_sample(theta1, Y, theta_fixed, M)
# all_weights_iid = ones(M) / M
# theta_new = MCEM_update(Y, all_Xs_iid, all_weights_iid)
# theta_old = theta1


# all_lik_increments = [complete_data_log_lik_increment(theta_new, theta_old, Y, X, theta_fixed) for X in all_Xs_iid]

# M = length(all_Xs_iid)

# A = dot(all_weights_iid, all_lik_increments)^2 
# A_iid = mean(all_lik_increments)^2

# B1 = dot(all_weights_iid.^2, all_lik_increments.^2)
# B2 = dot(all_weights_iid, all_lik_increments)^2
# B = B1 / B2
# B_iid = mean(all_lik_increments.^2) / (M * mean(all_lik_increments)^2)

# C1 = dot(all_weights_iid.^2, all_lik_increments)
# C2 = dot(all_weights_iid, all_lik_increments)
# C = 2 * C1 / C2
# C_iid = 2/M

# D = sum(all_weights_iid.^2)
# D_iid = 1/M

# ASE2 = A * (B - C + D) / M
# ASE2_iid = A_iid * (B_iid - C_iid + D_iid) / M

# ASE = sqrt(ASE2)
# ASE_iid = sqrt(ASE2_iid)



# get_resamp_ASE(theta_new, theta1, Y, all_Xs_iid, theta_fixed)
# get_ASE(theta_new, theta1, Y, all_Xs_iid, all_weights_iid, theta_fixed)


# ---------------------------------------------------------------------------- #
#             Various functions for within an ascent MCEM iteration            #
# ---------------------------------------------------------------------------- #

"""
Construct a lower confidence bound for improvement in the EM objective. Return true if this bound is positive.
Optionally returns the computed lower confidence bound.
"""
function check_ascent(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed, alpha; return_lcl = false)
    Q_increment = Q_MCEM_increment(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed)
    if Q_increment < 0
        error("Theta hat decreases MCEM objective.")
    end
    ASE = get_ASE(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed)
    multiplier = quantile(Normal(), 1 - alpha)

    lcl = Q_increment - multiplier*ASE

    if !return_lcl
        return lcl > 0
    else
        return lcl > 0, lcl
    end
end


"""
Construct an upper confidence bound for the EM increment. If smaller than the specified absolute tolerance, return true.
Optionally returns the computed upper confidence bound.
"""
function check_for_termination(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed, alpha, atol; diagnostics = false)
    Q_increment = Q_MCEM_increment(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed)
    ASE = get_ASE(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed)
    multiplier = quantile(Normal(), 1 - alpha)

    ucl = Q_increment + multiplier*ASE

    if !diagnostics
        return ucl < atol
    else
        return ucl < atol, ucl, ASE
    end
end



"""
Compute sample size for next iteration of MCEM.
Intermediate quantities are pre-computed.
"""
function get_next_MC_size(MCEM_increment, M_old, ASE, alpha1, alpha2)
    multiplier1 = quantile(Normal(), 1 - alpha1)
    multiplier2 = quantile(Normal(), 1 - alpha2)

    M_proposed = ceil(ASE^2 * (multiplier1 + multiplier2)^2 / MCEM_increment^2)

    M_new = max(M_proposed, M_old)

    return M_new
end


"""
Compute sample size for next iteration of MCEM.
Intermediate quantities are computed inside the function.
"""
function get_next_MC_size(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed, M_old, alpha1, alpha2)
    Q_increment = Q_MCEM_increment(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed)
    ASE = get_ASE(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed)

    return get_next_MC_size(Q_increment, M_old, ASE, alpha1, alpha2)
end





# ---------------------------------------------------------------------------- #
#                      Single ascent-based MCEM iteration                      #
# ---------------------------------------------------------------------------- #


"""
Object which contains the static parameters for MCEM
Note: M is excluded because it varies across iterations.
"""
mutable struct Ascent_MCEM_Control
    alpha1::Float64 # confidence level for checking whether to augment MC sample size
    alpha2::Float64 # confidence level for computing next step's initial MC sample size
    alpha3::Float64 # confidence level for checking whether to terminate MCEM
    k::Float64        # when augmenting MC sample, add M/k new points
    atol::Float64   # absolute tolerance for checking whether to terminate MCEM
end



"""
Perform a single iteration of ascent-based MCEM. Uses a level-alpha confidence bound to check for ascent. If not, augments the MC sample with M/k new points and tries again.
Options for diagnostics are included to check whether the MC sample was augmented.
"""
function ascent_MCEM_update(theta_old, Y, theta_fixed, M, alpha, k; return_MC_size = false, return_X=false, diagnostics = false)
    
    
    
    diagnostics = true
    
    
    
    
    
    all_Xs, all_raw_weights = get_importance_sample(theta_old, Y, theta_fixed, M; raw_weights=true)
    all_weights = normalize_weights(all_raw_weights)

    theta_new = MCEM_update(Y, all_Xs, all_weights)

    all_lcls = []

    if diagnostics
        ascended, lcl = check_ascent(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed, alpha; return_lcl=true)
        push!(all_lcls, lcl)
    else
        ascended = check_ascent(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed, alpha)
    end

    iter_count = 0

    while !ascended
        iter_count += 1
        this_samp_size = length(all_Xs)
        this_lcl = lcl
        Q_increment = Q_MCEM_increment(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed)
        this_ESS = round(get_ESS(all_weights), sigdigits=4)
        println("Inner Iteration: $iter_count, M = $this_samp_size, ESS = $this_ESS, Q Increment = $(round(Q_increment, sigdigits=2)), lcl = $(round(this_lcl, sigdigits=2)), Beta Hat: $(round(theta_new[1], sigdigits=3)), Sigma Hat: $(round(theta_new[2], sigdigits=3))")
        # println("Augmenting MC sample size...")
        new_Xs, new_raw_weights = get_importance_sample(theta_old, Y, theta_fixed, ceil(M/k), raw_weights=true)
        all_Xs = vcat(all_Xs, new_Xs)
        all_raw_weights = vcat(all_raw_weights, new_raw_weights)
        all_weights = normalize_weights(all_raw_weights)
        theta_new = MCEM_update(Y, all_Xs, all_weights)
        if diagnostics
            ascended, lcl = check_ascent(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed, alpha; return_lcl=true)
            push!(all_lcls, lcl)
        else
            ascended = check_ascent(theta_new, theta_old, Y, all_Xs, all_weights, theta_fixed, alpha)
        end
    end

    M_end = size(all_Xs, 1)






    diagnostics = false














    if diagnostics
        return M_end, all_lcls
    elseif !return_X && !return_MC_size
        return theta_new
    elseif return_X && !return_MC_size
        return theta_new, all_Xs, all_weights
    elseif !return_X && return_MC_size
        return theta_new, M_end
    else
        return theta_new, all_Xs, all_weights, M_end
    end
end


#* Check whether ascent-based MCEM algorithm augments the MC sample size.
# theta_hat_EM = run_EM(theta1, Y, theta_fixed)

# M = 10
# alpha = 0.4
# k = 3

# ascent_MCEM_update(theta_hat_EM, Y, theta_fixed, M, alpha, k; diagnostics = true)







"""
Perform a single iteration of ascent-based MCEM. Augments the MC size as necessary. Returns the initial MC size for next iteration, as well as a check for whether to terminate MCEM.
Note:   alpha1 - confidence level for checking whether to augment MC sample size
        alpha2 - confidence level for computing next step's initial MC sample size
        alpha3 - confidence level for checking whether to terminate MCEM

        k - when augmenting MC sample, add M/k new points
        atol - absolute tolerance for checking whether to terminate MCEM
"""
function full_ascent_MCEM_iteration(theta_old, Y, theta_fixed, M, alpha1, alpha2, alpha3, k, atol; return_X=false, diagnostics=false)

    theta_hat, all_Xs, all_weights = ascent_MCEM_update(theta_old, Y, theta_fixed, M, alpha1, k; return_X=true)
    M_end = size(all_Xs, 1)

    M_next = get_next_MC_size(theta_hat, theta_old, Y, all_Xs, all_weights, theta_fixed, M_end, alpha1, alpha2)

    ready_to_terminate, ucl, ASE = check_for_termination(theta_hat, theta_old, Y, all_Xs, all_weights, theta_fixed, alpha3, atol; diagnostics=true)

    if !return_X && !diagnostics
        return theta_hat, M_next, ready_to_terminate
    elseif return_X && !diagnostics
        return theta_hat, M_next, ready_to_terminate, all_Xs, all_weights
    elseif !return_X && diagnostics
        return theta_hat, M_next, ready_to_terminate, ucl, ASE
    else
        return theta_hat, M_next, ready_to_terminate, all_Xs, all_weights, ucl, ASE
    end
end





"""
Perform a single iteration of ascent-based MCEM. Augments the MC size as necessary. Returns the initial MC size for next iteration, as well as a check for whether to terminate MCEM.
Note:   alpha1 - confidence level for checking whether to augment MC sample size
        alpha2 - confidence level for computing next step's initial MC sample size
        alpha3 - confidence level for checking whether to terminate MCEM

        k - when augmenting MC sample, add M/k new points
        atol - absolute tolerance for checking whether to terminate MCEM
"""
function full_ascent_MCEM_iteration(theta_old, Y, theta_fixed, M, MCEM_control; return_X=false, diagnostics = false)

    # Unpack ascent-based MCEM parameters
    alpha1 = MCEM_control.alpha1
    alpha2 = MCEM_control.alpha2
    alpha3 = MCEM_control.alpha3
    k = MCEM_control.k
    atol = MCEM_control.atol

    return full_ascent_MCEM_iteration(theta_old, Y, theta_fixed, M, alpha1, alpha2, alpha3, k, atol; return_X=return_X, diagnostics=diagnostics)
end





# ---------------------------------------------------------------------------- #
#                       Full ascent-based MCEM algorithm                       #
# ---------------------------------------------------------------------------- #

"""
Run ascent-based MCEM algorithm.
Returns the final estimate of theta.
"""
function run_ascent_MCEM(theta_init, Y, theta_fixed, M_init, ascent_MCEM_control; diagnostics = false)

    # Initialize MCEM
    theta_hat = theta_init
    M = M_init
    ready_to_terminate = false

    iteration_count = 0

    # Run MCEM
    while !ready_to_terminate
        theta_hat, M, ready_to_terminate, ucl, ASE = full_ascent_MCEM_iteration(theta_hat, Y, theta_fixed, M, ascent_MCEM_control; diagnostics=true)
        iteration_count += 1
        # println(iteration_count)
        println("Outer Iteration: $iteration_count, MC size: $M, UCL = $(round(ucl, sigdigits=2)), ASE = $(round(ASE, sigdigits=2)), Beta Hat: $(round(theta_hat[1], sigdigits=3)), Sigma Hat: $(round(theta_hat[2], sigdigits=3))")
    end




    # if size(theta_hat, 1) == 1
    #     theta_hat = theta_hat
    # end

    if diagnostics
        return theta_hat, M
    else
        return theta_hat
    end
end



"""
Run Ascent-Based MCEM algorithm for B replications.
"""
function run_many_ascent_MCEMs(B, theta_init, theta_true, theta_fixed, M_init, ascent_MCEM_control)
        
    # Unpack theta_true
    beta_0, sigma_0 = theta_true

    # Unpack theta_fixed
    mu_0, tau_0 = theta_fixed

    # Container to store estimates of theta
    all_theta_hat_MCEMs = Vector{Vector{Float64}}(undef, B)

    # all_SE_hat_MCEMs = Vector{Vector{Float64}}(undef, B_MCEM)


    @showprogress for b in eachindex(all_theta_hat_MCEMs)
        # Generate data
        Random.seed!(b^2)
        this_X = rand(Normal(mu_0, tau_0), n)
        this_epsilon = rand(Normal(0, sigma_0), n)
        this_Y = beta_0 * this_X + this_epsilon

        # Estimate theta
        Random.seed!(b^2)
        this_theta_hat = run_ascent_MCEM(theta_init, this_Y, theta_fixed, M_init, ascent_MCEM_control)
        all_theta_hat_MCEMs[b] = this_theta_hat

        # # Estimate SE
        # #! Fix-SE
        # Random.seed!(b^2)
        # this_SE_hat = MCEM_SE_formula(this_theta_hat, this_Y, this_theta_hat, theta_fixed, M_SE)
        # all_SE_hat_MCEMs[b] = this_SE_hat

    end

    return all_theta_hat_MCEMs
end