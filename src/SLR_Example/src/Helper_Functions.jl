
export cov2cor
export get_eta
export get_ESS
export get_complete_cell_probs
export Y_from_X
export ispossemidef



"""
Converts a covariance matrix to the corresponding correlation matrix.
"""
function cov2cor(cov_mat)
    D = diagm(1 ./ sqrt.(abs.(diag(cov_mat))))
    return D * cov_mat * D
end


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



"""
Probabilities of each cell in the complete data model.
"""
function get_complete_cell_probs(theta0)
    p, q = theta0
    r = 1 - p - q

    return [r^2, 2*p*r, p^2, 2*q*r, q^2, 2*p*q]
end


"""
Sum the cells in X to get Y. I.e. Aggregate genotypes which have the same phenotype.
"""
function Y_from_X(X)
    return [X[1], X[2] + X[3], X[4] + X[5], X[6]]
end


"""
Checks whether the matrix A is positive semidefinite. Not fast.
"""
function ispossemidef(A)
    return all(eigvals(A) .>= 0)
end