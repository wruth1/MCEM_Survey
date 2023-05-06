
export num_O_alleles, num_A_alleles, num_B_alleles, num_OAB_alleles

export complete_data_log_lik
export complete_data_score
export complete_data_Hessian

export p_hat_complete, q_hat_complete
export complete_data_MLE


#ToDo - Adjust this function to work for X. I can then use it to analytically compute the standard error of the complete data MLE. Now that I'm thinking about it though, I'm not sure that's useful.
function Y_cov_mat(theta, Y)
    n = sum(Y)
    prob_vec = get_cell_probs(theta)

    A = n * diagm(prob_vec)
    B = n * prob_vec * prob_vec'

    output = A - B
    return output
end



# p_hat_vec = [0, 1, 2, 0, 0, 1] / (2*n)
# q_hat_vec = 

# ---------------------------------------------------------------------------- #
#                                Log-Likelihood                                #
# ---------------------------------------------------------------------------- #

# --- Functions which count the number of alleles of each type in a sample --- #

function num_O_alleles(X)
    return 2*X[1] + X[2] + X[4]
end

function num_A_alleles(X)
    return X[2] + 2*X[3] + X[6]
end

function num_B_alleles(X)
    return X[4] + 2*X[5] + X[6]
end

# Vector of allele counts in the order O, A, B
function num_OAB_alleles(X)
    num_O = num_O_alleles(X)
    num_A = num_A_alleles(X)
    num_B = num_B_alleles(X)

    return num_O, num_A, num_B
end


# ------------------------ Evaluate the log-likelihood ----------------------- #
function complete_data_log_lik(theta, Y, X)
    p, q = theta
    r = 1 - p - q

    num_O, num_A, num_B = num_OAB_alleles(X)

    output = num_O * log(r) + num_A * log(p) + num_B * log(q)    
    return output
end


# ---------------------------------------------------------------------------- #
#                                     Score                                    #
# ---------------------------------------------------------------------------- #


function complete_data_score(theta, Y, X)
    p, q = theta
    r = 1 - p - q

    num_O, num_A, num_B = num_OAB_alleles(X)

    g1 = num_A / p - num_O / r
    g2 = num_B / q - num_O / r

    return [g1, g2]
end





# ---------------------------------------------------------------------------- #
#                                    Hessian                                   #
# ---------------------------------------------------------------------------- #


function complete_data_Hessian(theta, Y, X)
    p, q = theta
    r = 1 - p - q

    num_O, num_A, num_B = num_OAB_alleles(X)

    h11 = -num_A / p^2 - num_O / r^2
    h12 = - num_O / r^2
    h22 = -num_B / q^2 - num_O / r^2

    return [h11 h12; h12 h22]
end





# ---------------------------------------------------------------------------- #
#                               Complete data MLE                              #
# ---------------------------------------------------------------------------- #

function p_hat_complete(Y, X)
    _, num_A, _ = num_OAB_alleles(X)
    n = sum(Y)

    return num_A / (2*n)
end

function q_hat_complete(Y, X)
    _, _, num_B = num_OAB_alleles(X)
    n = sum(Y)

    return num_B / (2*n)
end


function complete_data_MLE(Y, X)
    p_hat = p_hat_complete(Y, X)
    q_hat = q_hat_complete(Y, X)

    return [p_hat, q_hat]
end