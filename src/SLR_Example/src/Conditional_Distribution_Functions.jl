
export get_alpha1, get_alpha2, get_beta1, get_beta2
export mu_X_given_Y, cov_X_given_Y, mu2_X_given_Y
export cov_X_given_Y


# ---------------------------------------------------------------------------- #
#              Conditional probabilities of X compartments given Y             #
# ---------------------------------------------------------------------------- #

"""
Probability of genotype AA given the number of A phenotypes
"""
function get_alpha2(theta)
    p, q = theta
    r = 1 - p - q

    num = p^2
    den = num + 2 * p * r

    return num / den
end

"""
Probability of genotype A0 given the number of A phenotypes
"""
function get_alpha1(theta)
    return 1 - get_alpha2(theta)
end


"""
Probability of genotype BB given the number of B phenotypes
"""
function get_beta2(theta)
    p, q = theta
    r = 1 - p - q

    num = q^2
    den = num + 2 * q * r

    return num / den
end

"""
Probability of genotype B0 given the number of B phenotypes
"""
function get_beta1(theta)
    return 1 - get_beta2(theta)
end



# ---------------------------------------------------------------------------- #
#                       Conditional moments of X given Y                       #
# ---------------------------------------------------------------------------- #

"""
Conditional mean of X given Y
"""
function mu_X_given_Y(theta, Y)
    alpha1 = get_alpha1(theta)
    alpha2 = get_alpha2(theta)
    beta1 = get_beta1(theta)
    beta2 = get_beta2(theta)
    
    mu1 = Y[1]
    mu2 = alpha1 * Y[2] 
    mu3 = alpha2 * Y[2] 
    mu4 = beta1 * Y[3]
    mu5 = beta2 * Y[3]
    mu6 = Y[4]

    return [mu1, mu2, mu3, mu4, mu5, mu6]
end

"""
Conditional variance of X given Y
"""
function cov_X_given_Y(theta, Y)
    alpha1 = get_alpha1(theta)
    alpha2 = get_alpha2(theta)
    beta1 = get_beta1(theta)
    beta2 = get_beta2(theta)

    y2 = Y[2]
    y3 = Y[3]

    sigma2_1 = 0
    sigma2_2 = y2 * alpha1 * alpha2
    sigma2_3 = y2 * alpha1 * alpha2
    sigma2_4 = y3 * beta1 * beta2
    sigma2_5 = y3 * beta1 * beta2
    sigma2_6 = 0

    sigma_12 = 0
    sigma_13 = 0
    sigma_14 = 0
    sigma_15 = 0
    sigma_16 = 0

    sigma_23 = -y2 * alpha1 * alpha2
    sigma_24 = 0
    sigma_25 = 0
    sigma_26 = 0

    sigma_34 = 0
    sigma_35 = 0
    sigma_36 = 0

    sigma_45 = -y3 * beta1 * beta2
    sigma_46 = 0

    sigma_56 = 0

    Sigma = [sigma2_1 sigma_12 sigma_13 sigma_14 sigma_15 sigma_16;
             sigma_12 sigma2_2 sigma_23 sigma_24 sigma_25 sigma_26;
             sigma_13 sigma_23 sigma2_3 sigma_34 sigma_35 sigma_36;
             sigma_14 sigma_24 sigma_34 sigma2_4 sigma_45 sigma_46;
             sigma_15 sigma_25 sigma_35 sigma_45 sigma2_5 sigma_56;
             sigma_16 sigma_26 sigma_36 sigma_46 sigma_56 sigma2_6]

    return Sigma
end

"""
Conditional expectation of XX' given Y.
Output must be symmetric. We force this condition.
"""
function mu2_X_given_Y(theta, Y)
    Sigma = cov_X_given_Y(theta, Y)

    mu = mu_X_given_Y(theta, Y)
    mu_sq = mu * mu'

    mu2 = Sigma + mu_sq
    return Hermitian(mu2)
end
