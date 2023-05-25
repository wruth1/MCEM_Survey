
export get_ASE, get_resamp_ASE
export check_ascent, check_for_termination, get_next_MC_size
export Ascent_MCEM_Control
export ascent_MCEM_update, full_ascent_MCEM_iteration
export run_ascent_MCEM, run_many_ascent_MCEMs



#* ---------------------------------------------------------------------------- #
#*                                 IID Sampling                                 #
#* ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                 Asymptotic SE of increment in MCEM objective                 #
# ---------------------------------------------------------------------------- #



"""
Estimate the asymptotic standard error of the increment in MCEM objective function.
"""
function get_ASE(theta_new, theta_old, Y, all_Xs, all_weights)
    M = length(all_Xs)

    all_lik_increments = [complete_data_log_lik_increment(theta_new, theta_old, Y, X) for X in all_Xs]

    A = dot(all_weights, all_lik_increments)^2 
    
    B1 = dot(all_weights.^2, all_lik_increments.^2)
    B2 = dot(all_weights, all_lik_increments)^2
    B = B1 / B2

    C1 = dot(all_weights.^2, all_lik_increments)
    C2 = dot(all_weights, all_lik_increments)
    C = 2 * C1 / C2

    D = sum(all_weights.^2)

    ASE2 = A * (B - C + D)

    ASE = sqrt(ASE2)
    return ASE
end


"""
Estimate the asymptotic standard error of the increment in MCEM objective function directly from the log-likelihood increments.
Note: Also valid for iid sampling.
"""
function get_resamp_ASE(theta_new, theta_old, Y, all_resamp_Xs)

    # test_all_Xs = wsample(all_Xs, all_weights, M, replace = true)

    M = length(all_resamp_Xs)
    test_lik_increments = [complete_data_log_lik_increment(theta_new, theta_old, Y, X) for X in all_resamp_Xs]
    return std(test_lik_increments)/sqrt(M)

end

# ---------------------------------------------------------------------------- #
#                    Investigate ASE formula for IID sample                    #
# ---------------------------------------------------------------------------- #

# Random.seed!(1)

# all_Xs_iid, _ = get_importance_sample(theta1, Y, M)
# all_weights_iid = ones(M) / M
# theta_new = MCEM_update(Y, all_Xs_iid, all_weights_iid)
# theta_old = theta1


# all_lik_increments = [complete_data_log_lik_increment(theta_new, theta_old, Y, X) for X in all_Xs_iid]

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



# get_resamp_ASE(theta_new, theta1, Y, all_Xs_iid)
# get_ASE(theta_new, theta1, Y, all_Xs_iid, all_weights_iid)


# ---------------------------------------------------------------------------- #
#             Various functions for within an ascent MCEM iteration            #
# ---------------------------------------------------------------------------- #

"""
Construct a lower confidence bound for improvement in the EM objective. Return true if this bound is positive.
Optionally returns the computed lower confidence bound.
"""
function check_ascent(theta_new, theta_old, Y, all_Xs, alpha1; return_lcl = false)
    Q_increment = Q_MCEM_increment_iid(theta_new, theta_old, Y, all_Xs)
    if Q_increment < 0
        error("Theta hat decreases MCEM objective.")
    end
    ASE = get_resamp_ASE(theta_new, theta_old, Y, all_Xs)
    multiplier = quantile(Normal(), 1 - alpha1)

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
function check_for_termination(theta_new, theta_old, Y, all_Xs, alpha3, atol; diagnostics = false)
    Q_increment = Q_MCEM_increment_iid(theta_new, theta_old, Y, all_Xs)
    ASE = get_resamp_ASE(theta_new, theta_old, Y, all_Xs)
    multiplier = quantile(Normal(), 1 - alpha3)

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
function get_next_MC_size(theta_new, theta_old, Y, all_Xs, M_old, alpha1, alpha2)
    Q_increment = Q_MCEM_increment_iid(theta_new, theta_old, Y, all_Xs)
    ASE = get_resamp_ASE(theta_new, theta_old, Y, all_Xs)

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
function ascent_MCEM_update(theta_old, Y, M, alpha1, k; return_MC_size = false, return_X=false, diagnostics = false)
    
    
    
    diagnostics = true
    
    
    
    
    
    all_Xs = sample_X_given_Y_iid(M, theta_old, Y)

    theta_new = MCEM_update_iid(Y, all_Xs)

    all_lcls = []

    if diagnostics
        ascended, lcl = check_ascent(theta_new, theta_old, Y, all_Xs, alpha1; return_lcl=true)
        push!(all_lcls, lcl)
    else
        ascended = check_ascent(theta_new, theta_old, Y, all_Xs, alpha1)
    end

    iter_count = 0

    while !ascended
        iter_count += 1
        this_samp_size = length(all_Xs)
        this_lcl = lcl
        Q_increment = Q_MCEM_increment_iid(theta_new, theta_old, Y, all_Xs)
        println("Inner Iteration: $iter_count, M = $this_samp_size, Q Increment = $(round(Q_increment, sigdigits=2)), lcl = $(round(this_lcl, sigdigits=2))")

        new_Xs = sample_X_given_Y_iid(ceil(M/k), theta_old, Y)
        all_Xs = vcat(all_Xs, new_Xs)
        theta_new = MCEM_update_iid(Y, all_Xs)
        if diagnostics
            ascended, lcl = check_ascent(theta_new, theta_old, Y, all_Xs, alpha1; return_lcl=true)
            push!(all_lcls, lcl)
        else
            ascended = check_ascent(theta_new, theta_old, Y, all_Xs, alpha1)
        end
    end

    M_end = size(all_Xs, 1)






    diagnostics = false














    if diagnostics
        return M_end, all_lcls
    elseif !return_X && !return_MC_size
        return theta_new
    elseif return_X && !return_MC_size
        return theta_new, all_Xs
    elseif !return_X && return_MC_size
        return theta_new, M_end
    else
        return theta_new, all_Xs, M_end
    end
end


#* Check whether ascent-based MCEM algorithm augments the MC sample size.
# theta_hat_EM = run_EM(theta1, Y)

# M = 10
# alpha = 0.4
# k = 3

# ascent_MCEM_update(theta_hat_EM, Y, M, alpha, k; diagnostics = true)







"""
Perform a single iteration of ascent-based MCEM. Augments the MC size as necessary. Returns the initial MC size for next iteration, as well as a check for whether to terminate MCEM.
Note:   alpha1 - confidence level for checking whether to augment MC sample size
        alpha2 - confidence level for computing next step's initial MC sample size
        alpha3 - confidence level for checking whether to terminate MCEM

        k - when augmenting MC sample, add M/k new points
        atol - absolute tolerance for checking whether to terminate MCEM
"""
function full_ascent_MCEM_iteration(theta_old, Y, M, alpha1, alpha2, alpha3, k, atol; return_X=false, diagnostics=false)

    theta_hat, all_Xs = ascent_MCEM_update(theta_old, Y, M, alpha1, k; return_X=true)
    M_end = size(all_Xs, 1)

    M_next = get_next_MC_size(theta_hat, theta_old, Y, all_Xs, M_end, alpha1, alpha2)

    ready_to_terminate, ucl, ASE = check_for_termination(theta_hat, theta_old, Y, all_Xs, alpha3, atol; diagnostics=true)

    if !return_X && !diagnostics
        return theta_hat, M_next, ready_to_terminate
    elseif return_X && !diagnostics
        return theta_hat, M_next, ready_to_terminate, all_Xs
    elseif !return_X && diagnostics
        return theta_hat, M_next, ready_to_terminate, ucl, ASE
    else
        return theta_hat, M_next, ready_to_terminate, all_Xs, ucl, ASE
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
function full_ascent_MCEM_iteration(theta_old, Y, M, MCEM_control; return_X=false, diagnostics = false)

    # Unpack ascent-based MCEM parameters
    alpha1 = MCEM_control.alpha1
    alpha2 = MCEM_control.alpha2
    alpha3 = MCEM_control.alpha3
    k = MCEM_control.k
    atol = MCEM_control.atol

    return full_ascent_MCEM_iteration(theta_old, Y, M, alpha1, alpha2, alpha3, k, atol; return_X=return_X, diagnostics=diagnostics)
end




# ---------------------------------------------------------------------------- #
#                       Full ascent-based MCEM algorithm                       #
# ---------------------------------------------------------------------------- #

"""
Run ascent-based MCEM algorithm.
Returns the final estimate of theta.
"""
function run_ascent_MCEM(theta_init, Y, M_init, ascent_MCEM_control; diagnostics = false)

    # Initialize MCEM
    theta_hat = theta_init
    M = M_init
    ready_to_terminate = false

    iteration_count = 0

    all_theta_hats = []

    # Run MCEM
    while !ready_to_terminate
        theta_hat, M, ready_to_terminate, ucl, ASE = full_ascent_MCEM_iteration(theta_hat, Y, M, ascent_MCEM_control; diagnostics=true)
        push!(all_theta_hats, theta_hat)
        iteration_count += 1

        println("Outer Iteration: $iteration_count, MC size: $M, UCL = $(round(ucl, sigdigits=2)), ASE = $(round(ASE, sigdigits=2))")
    end




    # if size(theta_hat, 1) == 1
    #     theta_hat = theta_hat
    # end

    if diagnostics
        return theta_hat, all_theta_hats
    else
        return theta_hat
    end
end



"""
Run Ascent-Based MCEM algorithm for B replications.
"""
function run_many_ascent_MCEMs(B, theta_init, theta_true, M_init, ascent_MCEM_control)
        
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
        this_theta_hat = run_ascent_MCEM(theta_init, this_Y, M_init, ascent_MCEM_control)
        all_theta_hat_MCEMs[b] = this_theta_hat

        # # Estimate SE
        # #! Fix-SE
        # Random.seed!(b^2)
        # this_SE_hat = MCEM_SE_formula(this_theta_hat, this_Y, this_theta_hat, M_SE)
        # all_SE_hat_MCEMs[b] = this_SE_hat

    end

    return all_theta_hat_MCEMs
end






#* ---------------------------------------------------------------------------- #
#*                              Importance sampling                             #
#* ---------------------------------------------------------------------------- #