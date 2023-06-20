
export get_lik_update
export run_pilot_Chan_Ledolter
export process_pilot_Chan_Ledolter
export run_final_Chan_Ledolter
export run_MCEM_Chan_Ledolter



# M_pilot = 10
# K_max = 20

# R_keep = 5  # Numer of estimates to keep after maximizer in pilot study
# B = 5   # Number of replicate updates to take at each kept estimate

# delta = 0.01    # Allowable SE for estimate of observed data log lik rat

# using SLR_Example


# ---------------------------------------------------------------------------- #
#                           Functions for Pilot Study                          #
# ---------------------------------------------------------------------------- #

"""
Estimate the obs data log lik ratio from theta_old to theta_new, given an MC sample from theta_new.
Note: Actually estimate the ratio from theta_new to theta_old, then multiply by -1
"""
function get_lik_update(theta_new, theta_old, Y, all_Xs::Vector)
    all_log_increments = [complete_data_log_lik_increment(theta_old, theta_new, Y, X) for X in all_Xs]
    log_sum_increments = logsumexp(all_log_increments)
    return -(log_sum_increments - log(length(all_Xs)))
end

"""
Estimate the obs data log lik ratio from theta_old to theta_new, given an MC sample from theta_new.
Note: Actually estimate the ratio from theta_new to theta_old, then multiply by -1
"""
function get_lik_update(theta_new, theta_old, Y, M::Int)
    all_Xs = sample_X_given_Y_iid(M, theta_new, Y)
    return get_lik_update(theta_new, theta_old, Y, all_Xs)
end

"""
Run the pilot study procedure described in Chan and Ledolter (1995).
Returns the trajectory of estimates and estimated likelihood ratio relative to theta_init.
Note: log lik rat is estimated based on next iteration's MC sample. This makes the order of computation a little weird.
"""
function run_pilot_Chan_Ledolter(theta_init, Y, M_pilot, K_max)
    
    all_theta_hats = Vector(undef, K_max)
    all_log_lik_rats = Vector(undef, K_max)
    all_log_lik_updates = Vector(undef, K_max)

    theta_old = theta_init
    log_lik_rat = 0.0


    all_Xs = sample_X_given_Y_iid(M_pilot, theta_old, Y)

    for K in 1:K_max
        theta_new = MCEM_update_iid(Y, all_Xs)
        all_theta_hats[K] = theta_new

        all_Xs = sample_X_given_Y_iid(M_pilot, theta_new, Y)

        log_lik_update = get_lik_update(theta_new, theta_old, Y, all_Xs)
        all_log_lik_updates[K] = log_lik_update
        log_lik_rat += log_lik_update
        all_log_lik_rats[K] = log_lik_rat

        theta_old = theta_new
    end

    return all_theta_hats, all_log_lik_rats, all_log_lik_updates
end



# ---------------------------------------------------------------------------- #
#                                Run Pilot Study                               #
# ---------------------------------------------------------------------------- #

# all_theta_hats, all_log_lik_rats, all_log_lik_updates = run_pilot_Chan_Ledolter(theta_init, Y, 10000, K_max)


# plot(all_log_lik_updates, label="Log Likelihood Ratio")

# p_hat_traj = getindex.(all_theta_hats, 1)
# q_hat_traj = getindex.(all_theta_hats, 2)

# CL_plot = plot(p_hat_traj, label = "p", xlabel = "Iteration", ylabel = "Estimate", size=(800, 600), margin=10mm, legend=:right, legendfont=font(20), guidefont=font(20));
# plot!(CL_plot, q_hat_traj, label = "q");
# scatter!(CL_plot, p_hat_traj, label = nothing);
# scatter!(CL_plot, q_hat_traj, label = nothing);
# hline!(CL_plot, [theta_MLE], label = nothing, linewidth=2, linecolor=:black)




# ---------------------------------------------------------------------------- #
#                       Evaluate output from pilot study                       #
# ---------------------------------------------------------------------------- #

"""
From the pilot study, extract the best theta_hat and some nearby theta_hats, estimate the SE of the log lik ratio, and compute the final M which brings this SE below the specified tolerance, delta.
Returns the optimal theta_hat, the MC size which keeps our SE below the specified tolerance, and the corresponding MOE for the log lik ratio.
"""
function process_pilot_Chan_Ledolter(Y, all_theta_hats, all_log_lik_rats, M_pilot, R_keep, B, delta)

    ind_max_pilot = argmax(all_log_lik_rats)
    theta_max_pilot = all_theta_hats[ind_max_pilot]

    # ---------------- Extract  some theta_hats near the maximizer --------------- #

    # Some robustness is necessary here. Retain up to R_keep theta_hats, starting at the optimum and moving forward. Do not move backward.
    K_max = length(all_theta_hats)
    if ind_max_pilot + R_keep <= K_max
        theta_hats_pilot = all_theta_hats[ind_max_pilot:(ind_max_pilot + R_keep - 1)]
    else
        theta_hats_pilot = all_theta_hats[ind_max_pilot:K_max]
    end

    diagnostic_log_lik_updates = Vector(undef, length(theta_hats_pilot))
    for r in eachindex(theta_hats_pilot)
        this_log_lik_updates = Vector(undef, B)
        theta_hat = theta_hats_pilot[r]

        for b in 1:B
            theta_new = MCEM_update_iid(theta_hat, Y, M_pilot)
            this_update = get_lik_update(theta_new, theta_hat, Y, M_pilot)
            this_log_lik_updates[b] = this_update
        end

        diagnostic_log_lik_updates[r] = this_log_lik_updates
    end


    SE_log_lik_updates = sqrt(mean(std.(diagnostic_log_lik_updates)))


    M_final = ceil(Int, M_pilot * SE_log_lik_updates / delta)   # First argument converts output to integer

    SE_final = SE_log_lik_updates * M_pilot / M_final   # Asymptotics are order M, not order sqrt(M)

    # compute the 97.5th percentile of the standard normal distribution
    wald_mult = quantile(Normal(), 0.975)
    MOE_final =  wald_mult * SE_final 


    return theta_max_pilot, M_final, MOE_final, SE_log_lik_updates, SE_final
end


# ---------------------------------------------------------------------------- #
#                          Run MCEM until convergence                          #
# ---------------------------------------------------------------------------- #

"""
Iterate MCEM until the stopping rule of Chan and Ledolter (1995) is satisfied.
"""
function run_final_Chan_Ledolter(theta_init, Y, M_final, MOE_final)

    all_theta_hats = Vector()
    all_log_lik_rats = Vector()
    all_log_lik_updates = Vector()

    theta_old = theta_init
    log_lik_rat = 0.0

    all_Xs = sample_X_given_Y_iid(M_final, theta_old, Y)

    done = false

    while !done
        theta_new = MCEM_update_iid(Y, all_Xs)
        push!(all_theta_hats, theta_new)

        all_Xs = sample_X_given_Y_iid(M_final, theta_new, Y)

        log_lik_update = get_lik_update(theta_new, theta_old, Y, all_Xs)
        push!(all_log_lik_updates, log_lik_update)

        log_lik_rat += log_lik_update
        push!(all_log_lik_rats, log_lik_rat)

        if abs(log_lik_update) < MOE_final
            done = true
        else
            theta_old = theta_new
        end
    end

    return all_theta_hats, all_log_lik_rats, all_log_lik_updates
end



# ---------------------------------------------------------------------------- #
#                              Assemble the pieces                             #
# ---------------------------------------------------------------------------- #



# Y               # Observed data
# theta_init      # Starting value for theta

# M_pilot = 10    # MC size for MCEM iterations in pilot study
# K_max = 20      # Number of MCEM iterations to use in pilot study

# R_keep = 5      # Numer of estimates after maximizer to keep from pilot study
# B = 5           # Number of replicate updates to take at each kept estimate

# delta = 0.01    # Allowable SE for estimate of observed data log lik rat

function run_MCEM_Chan_Ledolter(Y, theta_init, M_pilot, K_max, R_keep, B, delta; diagnostics = false)
    all_theta_hats, all_log_lik_rats, _ = run_pilot_Chan_Ledolter(theta_init, Y, M_pilot, K_max)

    theta_max_pilot, M_final, MOE_final, SE_pilot, SE_final = process_pilot_Chan_Ledolter(Y, all_theta_hats, all_log_lik_rats, M_pilot, R_keep, B, delta)

    all_final_theta_hats, all_final_log_lik_rat_updates, _ = run_final_Chan_Ledolter(theta_max_pilot, Y, M_final, MOE_final)
    all_final_log_lik_rats = all_log_lik_rats[end] .+ all_final_log_lik_rat_updates

    push!.(Ref(all_theta_hats), all_final_theta_hats)
    push!.(Ref(all_log_lik_rats), all_final_log_lik_rats)
    # push!.(Ref(all_log_lik_updates), all_final_log_lik_updates)

    if diagnostics
        return all_theta_hats, all_log_lik_rats, SE_pilot, SE_final
    else
        return all_theta_hats[end]
    end
end


# theta_hat_traj_CL, lik_rat_traj_CL = run_MCEM_Chan_Ledolter(Y, theta_init, M_pilot, K_max, R_keep, B, delta; diagnostics=true)