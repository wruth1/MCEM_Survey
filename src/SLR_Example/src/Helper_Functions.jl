
export get_eta
export get_ESS

"""
Compute the marginal variance of Y for the given value of beta. I.e. Return tau^2 * beta^2 + sigma^2.
"""
function get_eta(theta, theta_fixed)
    # Unpack theta
    beta, sigma = theta

    # Unpack theta_fixed
    mu, tau = theta_fixed

    return tau^2 * beta^2 + sigma^2
end


"""
Compute effective sample size from the given set of weights.
"""
function get_ESS(weights)
    return 1 / sum(weights.^2)
end


