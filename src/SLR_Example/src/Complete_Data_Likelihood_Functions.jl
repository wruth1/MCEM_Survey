
export complete_data_log_lik
export complete_data_score_term, complete_data_score
export complete_data_Hessian_term, complete_data_Hessian



# ---------------------------------------------------------------------------- #
#                                Log-Likelihood                                #
# ---------------------------------------------------------------------------- #


function complete_data_log_lik(theta, Y, X, theta_fixed)
    # Unpack theta
    beta, sigma = theta

    # Unpack theta_fixed
    mu, tau = theta_fixed

    eta = get_eta(theta, theta_fixed)

    # Define parameters of MVNormal
    mu_vec = [mu * beta, mu]
    cov_mat = [eta beta * tau^2; beta * tau^2 tau^2]

    # Compute log-likelihood
    output = 0
    for i in eachindex(Y)
        output += logpdf(MvNormal(mu_vec, cov_mat), [Y[i], X[i]])
    end
    
    return output
end


# ---------------------------------------------------------------------------- #
#                                     Score                                    #
# ---------------------------------------------------------------------------- #

function complete_data_score_term(theta, y, x, theta_fixed)
    # Unpack theta
    beta, sigma = theta

    # Unpack theta_fixed
    mu, tau = theta_fixed

    # ------------------------------- Compute score ------------------------------ #

    # beta derivative
    A1 = y * x
    A2 = - beta * x^2

    A = (A1 + A2) / sigma^2

    # sigma derivative
    B1 = -1/sigma
    B2 = (y - x*beta)^2 / (sigma^3)

    B = B1 + B2

    return [A, B]
end



function complete_data_score(theta, Y, X, theta_fixed)
    output = [0, 0]
    for i in eachindex(Y)
        output += complete_data_score_term(theta, Y[i], X[i], theta_fixed)
    end
    return output
end





# ---------------------------------------------------------------------------- #
#                                    Hessian                                   #
# ---------------------------------------------------------------------------- #

function complete_data_Hessian_term(theta, y, x, theta_fixed)
    # Unpack theta
    beta, sigma = theta

    # Unpack theta_fixed
    mu, tau = theta_fixed


    # ------------------------------- Compute Hessian ------------------------------ #

    # Second order beta derivative
    A = -x^2 / sigma^2

    # Cross derivative
    B1 = - 2 * x / sigma^3
    B2 = (y - x*beta)
    B = B1 * B2

    # Second order sigma derivative
    C1 = 1/sigma^2
    C2 = 3 / sigma^4
    C3 = (y - x*beta)^2
    C = C1 - C2 * C3

    return [A B; B C]
end



function complete_data_Hessian(theta, Y, X, theta_fixed)
    output = [0 0; 0 0]
    for i in eachindex(Y)
        output += complete_data_Hessian_term(theta, Y[i], X[i], theta_fixed)
    end
    return output
end

