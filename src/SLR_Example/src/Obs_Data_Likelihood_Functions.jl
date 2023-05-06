
export obs_data_log_lik
export obs_data_score1, obs_data_score2, obs_data_score
export obs_data_Hessian1, obs_data_Hessian2, obs_data_Hessian3, obs_data_Hessian
export obs_data_MLE
export obs_data_obs_info, obs_data_MLE_Cov, obs_data_MLE_SE

# using Optim

# theta = [0.2, 0.4]
# Y = [176, 182, 60, 17]



# sum(get_cell_probs(theta))

# ---------------------------------------------------------------------------- #
#                                log-likelihood                                #
# ---------------------------------------------------------------------------- #

# ------------------- First, define each cell's probability ------------------ #


function prob_cell1(theta)
    p, q = theta
    r = 1 - p - q
    return r^2
end

function prob_cell2(theta)
    p, q = theta
    r = 1 - p - q
    return p^2 + 2*p*r
end

function prob_cell3(theta)
    p, q = theta
    r = 1 - p - q
    return q^2 + 2*q*r
end

function prob_cell4(theta)
    p, q = theta
    r = 1 - p - q
    return 2*p*q
end

function get_cell_probs(theta)
    return [prob_cell1(theta), prob_cell2(theta), prob_cell3(theta), prob_cell4(theta)]
end


# ---------------------- Now, compute the log-likelihood --------------------- #

function obs_data_log_lik(theta, Y)
    p, q = theta
    r = 1 - p - q

    probs = get_cell_probs(theta)
    log_probs = log.(probs)

    return sum(Y .* log_probs)
end


# ---------------------------------------------------------------------------- #
#                                     score                                    #
# ---------------------------------------------------------------------------- #

# --------- First, we compute the gradient of each cell's probability -------- #

function grad_cell1(theta)
    p, q = theta
    r = 1 - p - q

    return [-2r, -2r]
end

function grad_cell2(theta)
    p, q = theta
    r = 1 - p - q

    return [2r, -2p]
end

function grad_cell3(theta)
    p, q = theta
    r = 1 - p - q

    return [-2q, 2r]
end

function grad_cell4(theta)
    p, q = theta
    r = 1 - p - q

    return [2q, 2p]
end

function get_cell_grads(theta)
    return [grad_cell1(theta), grad_cell2(theta), grad_cell3(theta), grad_cell4(theta)]
end


# -------------- Now, compute the gradient of the log-likelihood ------------- #

function obs_data_score(theta, Y)

    grads = get_cell_grads(theta)
    probs = get_cell_probs(theta)

    return sum(Y .* grads .* (1 ./ probs))
end


# ---------------------------------------------------------------------------- #
#                                    Hessian                                   #
# ---------------------------------------------------------------------------- #

# As before, first we give the second derivative of each cell's probability

function hess_cell1()
    return [2 2; 2 2]    
end

function hess_cell2()
    return [-2 -2; -2 0]
end

function hess_cell3()
    return [0 -2; -2 -2]
end

function hess_cell4()
    return [0 2; 2 0]
end

function get_cell_hessians()
    return [hess_cell1(), hess_cell2(), hess_cell3(), hess_cell4()]
end


# -------- Next, we need the outer product of each gradient with itself ------- #
function get_cell_grads_squared(theta)
    grads = get_cell_grads(theta)
    outer_prods = [grad * grad' for grad in grads]
    
    return outer_prods
end


# ------------- Now, we compute the Hessian of the log-likelihood ------------ #
function obs_data_Hessian(theta, Y)
    cell_probs = get_cell_probs(theta)
    cell_grad_prods = get_cell_grads_squared(theta)
    cell_hessians = get_cell_hessians()

    A = cell_probs .* cell_hessians
    B = cell_grad_prods
    C = cell_probs .^ 2

    output = sum(Y .* (A .- B) ./ C)
    return output
end



# ---------------------------------------------------------------------------- #
#                                      MLE                                     #
# ---------------------------------------------------------------------------- #

"""
Find the value of theta (i.e. p and q) which maximizes the observed data log-likelihood.
Performs numerical optimization using the BFGS algorithm. Optionally, supply an initial value of theta.
"""
function obs_data_MLE(Y; theta_init = [1/3, 1/3])
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, 0 <= p_jump <= 1, start = theta_init[1])
    @variable(model, 0 <= q_jump <= 1, start = theta_init[2])
    this_obj_fun(p_jump, q_jump) = obs_data_log_lik([p_jump, q_jump], Y)
    function this_grad!(g, p_in, q_in)
        grad_val = obs_data_score([p_in, q_in], Y)
        g[1] = grad_val[1]
        g[2] = grad_val[2]
    end
    function this_hess!(H, p_in, q_in)
        hess_val = obs_data_Hessian([p_in, q_in], Y)
        H[1, 1] = hess_val[1, 1]
        H[1, 2] = hess_val[1, 2]
        H[2, 1] = hess_val[2, 1]
        H[2, 2] = hess_val[2, 2]
    end
    register(model, :this_obj_fun, 2, this_obj_fun, this_grad!)
    @NLobjective(model, Max, this_obj_fun(p_jump, q_jump))
    @constraint(model, p_jump + q_jump <= 1)
    optimize!(model)

    p_hat = value(p_jump)
    q_hat = value(q_jump)
    theta_hat = [p_hat, q_hat]
    return theta_hat
end


# --------------------------- Standard Error of MLE -------------------------- #

"""
Evaluate the observed information (negative Hessian) of the observed data likelihood at the MLE.
MLE provided as an argument.
"""
function obs_data_obs_info(theta_hat, Y)
    Hessian = obs_data_Hessian(theta_hat, Y)
    return -Hessian
end


"""
Estimate the covariance matrix of the MLE using the obs data information matrix.
"""
function obs_data_MLE_Cov(theta_hat, Y)
    obs_info = obs_data_obs_info(theta_hat, Y)
    return inv(obs_info)
end

"""
Estimate the standard error of the MLE using the obs data information matrix.
"""
function obs_data_MLE_SE(theta_hat, Y)
    MLE_Cov = obs_data_MLE_Cov(theta_hat, Y)
    return sqrt.(diag(MLE_Cov))
end

obs_data_obs_info(theta_hat, Y)
obs_data_MLE_Cov(theta_hat, Y)

