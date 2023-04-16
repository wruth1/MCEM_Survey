
export mu_X_given_Y, var_X_given_Y
export mu2_X_given_Y, mu3_X_given_Y, mu4_X_given_Y


# --------------------- Conditional moments of X given Y --------------------- #

"""
Conditional expectation of X given Y=y.
"""
function mu_X_given_Y(theta, y, theta_fixed)
    # Unpack theta
    beta, sigma = theta

    # Unpack theta_fixed
    mu, tau = theta_fixed    
    
    A = beta * tau^2 / get_eta(theta, theta_fixed)
    B = y - mu * beta

    output = mu + A * B
    return output
end


"""
Conditional variance of X given Y=y.
Note: This quantity does not depend on the value y.
"""
function var_X_given_Y(theta, theta_fixed)
    # Unpack theta
    beta, sigma = theta

    # Unpack theta_fixed
    mu, tau = theta_fixed
        
    A = beta^2 * tau^4 / get_eta(theta, theta_fixed)
    
    output = tau^2 - A
    return output
end

"""
Second conditional moment of X given Y=y.
"""
function mu2_X_given_Y(theta, y, theta_fixed)
    A = var_X_given_Y(theta, theta_fixed)
    B = mu_X_given_Y(theta, y, theta_fixed)

    return A + B^2
end


"""
Third conditional moment of X given Y=y.
"""
function mu3_X_given_Y(theta, y, theta_fixed)
    A = var_X_given_Y(theta, theta_fixed)
    B = mu_X_given_Y(theta, y, theta_fixed)

    return B^3 + 3 * A * B
end

"""
Fourth conditional moment of X given Y=y.
"""
function mu4_X_given_Y(theta, y, theta_fixed)
    A = var_X_given_Y(theta, theta_fixed)
    B = mu_X_given_Y(theta, y, theta_fixed)

    return B^4 + 6 * A * B^2 + 3 * A^2
end