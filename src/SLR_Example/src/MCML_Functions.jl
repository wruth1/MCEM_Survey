
export run_MCML

"""
Normalize a list of numbers to sum to 1 given the logs of those numbers. I.e. Normalize the exponentials of the inputs.
"""
function norm_exp(all_logs)
    log_norm_const = logsumexp(all_logs)

    return exp.(all_logs .- log_norm_const)
end

"""
Compute the log of the mean likelihood ratio for a given theta.
"""
function MCML_obj(theta, theta_ref, Y, all_Xs)
    all_log_lik_rats = [complete_data_log_lik_increment(theta, theta_ref, Y, X) for X in all_Xs]
    log_sum_lik_rats = logsumexp(all_log_lik_rats)
    log_mean_lik_rat = log_sum_lik_rats - log(length(all_Xs))
    return log_mean_lik_rat
end


"""
Setup and run an optimization problem to maximize the mean complete-data likelihood ratio.
"""
function optimize_MCML_obj(theta_ref, Y, all_Xs)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, 0 <= p_jump <= 1, start = theta_ref[1])
    @variable(model, 0 <= q_jump <= 1, start = theta_ref[2])
    this_obj_fun(p_jump, q_jump) = MCML_obj([p_jump, q_jump], theta_ref, Y, all_Xs)
    # Set gradient
    function grad!(G, p_in, q_in)
        theta_in = [p_in, q_in]
        all_scores = [complete_data_score(theta_in, Y, X) for X in all_Xs]
        
        all_log_lik_rats = [complete_data_log_lik_increment(theta_in, theta_ref, Y, X) for X in all_Xs]
        all_norm_lik_rats = norm_exp(all_log_lik_rats)

        new_grad = sum(all_scores .* all_norm_lik_rats)
        G[1] = new_grad[1]
        G[2] = new_grad[2]
    end
    
    register(model, :this_obj_fun, 2, this_obj_fun, grad!)
    @NLobjective(model, Max, this_obj_fun(p_jump, q_jump))
    @constraint(model, p_jump + q_jump <= 1)
    optimize!(model)

    p_hat = value(p_jump)
    q_hat = value(q_jump)
    theta_hat = [p_hat, q_hat]
    return theta_hat
end


"""
Run the Monte Carlo maximum likelihood algorithm.
"""
function run_MCML(theta_ref, Y, M)

    all_Xs = sample_X_given_Y_iid(M, theta_ref, Y)
    theta_hat = optimize_MCML_obj(theta_ref, Y, all_Xs)
    return theta_hat    

end