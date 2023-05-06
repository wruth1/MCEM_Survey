
export get_alpha1, get_alpha2, get_beta1, get_beta2
export mu_X_given_Y


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