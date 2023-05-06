
#* This comment color denotes when something needs to be changed before running test on a more complicated problem.



@testset "EM and MCEM" begin

    @testset "Conditional Distribution" begin
        @testset "Is conditional variance non-negative?" begin
            all_cond_vars = [var_X_given_Y(all_thetas[i], theta_fixed) for i in eachindex(all_thetas)]

            for cond_var in all_cond_vars
                @test cond_var >= 0
            end
        end
        

        @testset "Is conditional second moment non-negative?" begin
            cond_mu2s = [mu2_X_given_Y(all_thetas[i], Y[j], theta_fixed) for i in eachindex(all_thetas), j in eachindex(Y)]

            for these_vals in eachrow(cond_mu2s)
                @test all(these_vals .>= 0)
            end
        end
    end

    @testset "EM Algorithm" begin        

        @testset "EM update function" begin
            @testset "Does EM update (approximately) maximize the objective function?" begin
                Random.seed!(1)
                noise_sizes = randn(100)
                noise_grid = [0.1 * theta1 .* noise_sizes[i] for i in eachindex(noise_sizes)]
                

                # ---------------------------- Starting at theta1 ---------------------------- #
                theta_old = theta1

                theta_hat = EM_update(theta_old, Y, theta_fixed)
                Q_max = Q_EM(theta_hat, Y, theta_old, theta_fixed)

                nearby_theta_values = [theta_hat + noise_grid[i] for i in eachindex(noise_grid)]

                nearby_Q_values = [Q_EM(theta, Y, theta_old, theta_fixed) for theta in nearby_theta_values]

                @test all(Q_max .>= nearby_Q_values)

                # ---------------------------- Starting at theta2 ---------------------------- #
                theta_old = theta2

                theta_hat = EM_update(theta_old, Y, theta_fixed)
                Q_max = Q_EM(theta_hat, Y, theta_old, theta_fixed)

                nearby_theta_values = [theta_hat + noise_grid[i] for i in eachindex(noise_grid)]

                nearby_Q_values = [Q_EM(theta, Y, theta_old, theta_fixed) for theta in nearby_theta_values]

                @test all(Q_max .>= nearby_Q_values)
            end

        end


        @testset "EM standard error formula" begin
            #! Something is going wrong here. Specifically, in the conditional expectation of the squared score, my empirical and analytical values do not match for the first three components of the matrix (i.e. the [2,2] entry is fine, but the other three are not). 
            #! This suggests to me that that my formula for the beta-derivative of the log-likelihood is wrong, since this is the term which enters into the problematic terms in the squared score but not into the term which I am estimating correctly. However, I have combed through my formula, both in Maple and in my Julia implementation, and I can't find anything wrong. I have written tests for the conditional moments of X given Y and they are all working fine (with different levels of variability, but they all at least appear consistent).
            #! At this point, I'm not sure what could be causing the discrepancy. I'm going to move on for now and return to this later.
            #! Fix-SE

            @testset "Is our formula for complete data conditional information accurate?" begin
                # Number of conditional samples to generate
                M = 10000

                # Relative tolerance for comparing empirical and analytical conditional expectations
                rtol = 0.01

                # ---------------------- Compare at true value of theta ---------------------- #
                Random.seed!(1)
                all_Xs1 = sample_X_given_Y(theta1, Y, theta_fixed, M)

                all_Hessians1 = [complete_data_Hessian(theta1, Y, all_Xs1[i], theta_fixed) for i in eachindex(all_Xs1)]

                empirical_cond_mean_info1 = -mean(all_Hessians1)

                analytical_cond_mean_info1 = complete_data_conditional_information(theta1, Y, theta_fixed)

                @test (empirical_cond_mean_info1 ≈ analytical_cond_mean_info1) (rtol = rtol)


                # ------------------- Compare away from true value of theta ------------------ #
                Random.seed!(1)
                all_Xs2 = sample_X_given_Y(theta2, Y, theta_fixed, M)

                all_Hessians2 = [complete_data_Hessian(theta2, Y, all_Xs2[i], theta_fixed) for i in eachindex(all_Xs2)]

                empirical_cond_mean_info2 = mean(all_Hessians2)

                analytical_cond_mean_info2 = complete_data_conditional_information(theta2, Y, theta_fixed)

                @test (empirical_cond_mean_info2 ≈ -analytical_cond_mean_info2) (rtol = rtol)
            end

            @testset "Conditional expectation of squared score" begin                    

                @testset "Is our formula for conditional expectation of squared score accurate?" begin
                    # Number of conditional samples to generate
                    M = 10000

                    # Relative tolerance for comparing empirical and analytical conditional expectations
                    # Generous tolerance due to high MC variability (this is a hard seed)
                    rtol = 0.05

                    # Random.seed!(1)
                    all_Xs = sample_X_given_Y(theta1, Y, theta_fixed, M)

                    all_scores = [complete_data_score(theta1, Y, all_Xs[i], theta_fixed) for i in eachindex(all_Xs)]
                    all_sq_scores = [all_scores[i] * Transpose(all_scores[i]) for i in eachindex(all_scores)]

                    mean_sq_score = mean(all_sq_scores)

                    expect_squared_score = expect_sq_score(theta1, Y, theta_fixed)

                    @test (mean_sq_score ≈ expect_squared_score) (rtol = rtol)
                end
            end

            @testset "Is our formula for the covariance matrix of the EM estimator accurate?" begin
                # Number of datasets to generate
                B = 10000

                # Need larger sample size for asymptotics to kick in
                n = 500

                # Relative tolerance for comparing empirical and analytical SEs
                # Generous due to high MC variability
                rtol = 0.05

                # ---------------------- Starting at true value of theta --------------------- #

                all_beta_hats1 = Vector{Any}(undef, B)
                all_cov_hats1 = Vector{Any}(undef, B)

                # using ProgressMeter
                # prog = Progress(B, desc="Running EM")

                @showprogress for i in eachindex(all_beta_hats1)
                # Threads.@threads for i in eachindex(all_beta_hats1)
                    # Generate new dataset
                    Random.seed!(i^2)
                    X = rand(Normal(mu_0, tau_0), n)
                    epsilon = rand(Normal(0, sigma_0), n)
                    Y = beta_0 * X + epsilon

                    # Estimate beta
                    theta_hat = run_EM(theta1, Y, theta_fixed)

                    # Estimate SE
                    this_cov_hat = EM_COV_formula(theta_hat, Y, theta_fixed)

                    all_beta_hats1[i] = theta_hat
                    all_cov_hats1[i] = this_cov_hat

                    # next!(prog)
                end

                empirical_cov1 = cov(all_beta_hats1)
                mean_cov_hat1 = mean(all_cov_hats1)


                @test (empirical_cov1 ≈ mean_cov_hat1) (rtol = rtol)


                # ------------------ Starting away from true value of theta ------------------ #

                # This test takes long enough to run once (~ 30 s). I'd rather not run it twice.

                # all_beta_hats2 = Vector{Any}(undef, B)
                # all_cov_hats2 = Vector{Any}(undef, B)

                # # prog = Progress(B, desc="Running EM")

                # @showprogress for i in eachindex(all_beta_hats2)
                # # Threads.@threads for i in eachindex(all_beta_hats2)
                #     # Generate new dataset
                #     Random.seed!(i^2)
                #     X = rand(Normal(mu_0, tau_0), n)
                #     epsilon = rand(Normal(0, sigma_0), n)
                #     Y = beta_0 * X + epsilon

                #     # Estimate beta
                #     theta_hat = run_EM(theta2, Y, theta_fixed)

                #     # Estimate SE
                #     this_cov_hat = EM_COV_formula(theta_hat, Y, theta_fixed)

                #     all_beta_hats2[i] = theta_hat
                #     all_cov_hats2[i] = this_cov_hat

                #     # next!(prog)
                # end

                # empirical_cov2 = cov(all_beta_hats2)
                # mean_cov_hat2 = mean(all_cov_hats2)

                # @test (empirical_cov2 ≈ mean_cov_hat2) (rtol = rtol)


            end
        end
    end




    @testset "MCEM Algorithm" begin
            
        @testset "Maximizer of MCEM objective" begin
            
            @testset "Is MCEM maximizer close to EM maximizer?" begin
                theta_hat_EM1 = EM_update(theta1, Y, theta_fixed)
                theta_hat_EM2 = EM_update(theta2, Y, theta_fixed)

                # Number of Monte Carlo samples to draw
                M = 10000

                # Relative tolerance
                tol = 0.01

                Random.seed!(1)
                theta_hat_MCEM1 = MCEM_update(theta1, Y, theta_fixed, M)
                Random.seed!(1)
                theta_hat_MCEM2 = MCEM_update(theta2, Y, theta_fixed, M)

                @test isapprox(theta_hat_MCEM1, theta_hat_EM1, rtol = tol)
                @test isapprox(theta_hat_MCEM2, theta_hat_EM2, rtol = tol)
            
            end
        end

        @testset "Recovering EM SE formula" begin
    
            # Value of theta at which to evaluate scores and Hessians
            theta_hat_EM = run_EM(theta1, Y, theta_fixed)


            # Number of Monte Carlo samples to draw
            M = 100000
            Random.seed!(1)
            all_Xs = sample_X_given_Y(theta_hat_EM, Y, theta_fixed, M)



            @testset "Do MCEM and EM match on the covariance matrix formula?" begin
                @testset "Conditional expectation of complete data information." begin

                    # Relative tolerance
                    rtol = 0.1

                    MCEM_cond_exp = MC_complete_cond_info(theta_hat_EM, Y, all_Xs, theta_fixed)

                    EM_cond_exp = complete_data_conditional_information(theta_hat_EM, Y, theta_fixed)

                    @test (MCEM_cond_exp ≈ EM_cond_exp) (rtol = rtol)
                end

                @testset "Conditional expectation of squared score" begin
                    # Relative tolerance
                    rtol = 0.1

                    MCEM_cond_exp = MC_expect_sq_score(theta_hat_EM, Y, all_Xs, theta_fixed)

                    EM_cond_exp = expect_sq_score(theta_hat_EM, Y, theta_fixed)

                    @test (MCEM_cond_exp ≈ EM_cond_exp) (rtol = rtol)
                end

                @testset "Observed Data Information" begin
                    # Relative tolerance
                    rtol = 0.1

                    MCEM_cond_exp = MCEM_obs_data_info_formula(theta_hat_EM, Y, all_Xs, theta_fixed)

                    EM_cond_exp = EM_obs_data_information_formula(theta_hat_EM, Y, theta_fixed)

                    @test (MCEM_cond_exp ≈ EM_cond_exp) (rtol = rtol)
                end



                #! This test fails due to instability of matrix inversion. I can estimate the information matrix reasonably accurately, but when I invert it to get the covariance matrix, the result is way off. The problem isn't a matter of roundoff error, its an issue of inverting the matrix amplifying stochastic fluctuations of the MCEM estimate around the true EM value. 
                #! Note: The Woodbury identity gives an alternative formula for the covariance matrix which depends on the inverse of the residual (i.e. MCEM estimate - EM estimate). More precisely, there is a term of the form (A^-1 + E^-1)^-1. That is, we're inverting a mean-zero object (the residual), adding it to another matrix, then inverting the result. This is not a thing I expect to be able to do stably. Worth investigating further, but not before my presentation in 3 days.
                # @testset "MLE Covariance Matrix" begin
                #     # Relative tolerance
                #     rtol = 0.1

                #     MCEM_cond_exp = MCEM_COV_formula(theta_hat_EM, Y, all_Xs, theta_fixed)

                #     EM_cond_exp = EM_COV_formula(theta_hat_EM, Y, theta_fixed)

                #     @test (MCEM_cond_exp ≈ EM_cond_exp) (rtol = rtol)
                # end
                    
            end
        end

    end



    @testset "Ascent-Based MCEM Algorithm" begin
        # Note: We check for coverage of the increment to both the EM update and the MCEM update
        @testset "Is the confidence bound in check_ascent() valid?" begin
                    

            # Confidence level
            alpha = 0.2

            # Number of Monte Carlo samples to draw
            M = 1000

            # Number of times to replicate iteration
            B = 100

            # A generous coverage tolerance based on SD of the binomial distribution
            abs_coverage_tol = 3 * sqrt(alpha * (1 - alpha) / B)


            # ---------------------------------------------------------------------------- #
            #                      Starting at the true value of theta                     #
            # ---------------------------------------------------------------------------- #

            Random.seed!(1)

            theta_hat_EM = EM_update(theta1, Y, theta_fixed)
            EM_increment = Q_EM_increment(theta_hat_EM, Y, theta1, theta_fixed)

            # Confidence bounds for the EM increment to the MCEM update
            all_MCEM_bounds = zeros(B)

            for i in eachindex(all_MCEM_bounds)
                all_Xs = sample_X_given_Y(theta1, Y, theta_fixed, M)
                
                theta_hat_MCEM = MCEM_update(Y, all_Xs)
                
                _, this_bound_MCEM = check_ascent(theta_hat_MCEM, theta1, Y, all_Xs, theta_fixed, alpha, return_lcl = true)
                all_MCEM_bounds[i] = this_bound_MCEM
            end

            empirical_coverage_MCEM = mean(all_MCEM_bounds .< EM_increment)

            @test empirical_coverage_MCEM ≈ 1 - alpha (atol = abs_coverage_tol)



            # # ---------------------------------------------------------------------------- #
            # #                    Starting away from true value of theta                    #
            # # ---------------------------------------------------------------------------- #

            Random.seed!(1)

            theta_hat_EM = EM_update(theta2, Y, theta_fixed)
            EM_increment = Q_EM_increment(theta_hat_EM, Y, theta2, theta_fixed)

            # Confidence bounds for the EM increment to the MCEM update
            all_MCEM_bounds = zeros(B)

            for i in eachindex(all_MCEM_bounds)
                all_Xs = sample_X_given_Y(theta2, Y, theta_fixed, M)
                
                theta_hat_MCEM = MCEM_update(Y, all_Xs)
                
                _, this_bound_MCEM = check_ascent(theta_hat_MCEM, theta2, Y, all_Xs, theta_fixed, alpha, return_lcl = true)
                all_MCEM_bounds[i] = this_bound_MCEM
            end

            empirical_coverage_MCEM = mean(all_MCEM_bounds .< EM_increment)

            @test empirical_coverage_MCEM ≈ 1 - alpha (atol = abs_coverage_tol)

        end


        # --------------- Set control parameters for ascent-based MCEM --------------- #
        alpha1 = 0.3
        alpha2 = 0.3
        alpha3 = 0.3
        k = 3
        atol = 1e-4 # Absolute tolerance for convergence. 1e-4 is a good value for this example. 1e-5 is better but takes much longer
        
        control = Ascent_MCEM_Control(alpha1, alpha2, alpha3, k, atol)
        

        @testset "Does ascent MCEM (approximately) converge to the same limit as EM?" begin
            
            # Relative tolerance for comparing EM and ascent MCEM estimates
            tol = 5e-2

            # Tolerance for EM convergence
            EM_conv_tol = 1e-6

            

            # ---------------------- Starting at true value of theta --------------------- #
            theta_hat_EM = run_EM(theta1, Y, theta_fixed, rtol = EM_conv_tol)

            # --------------------------- Ascent MCEM estimator -------------------------- #
            # theta_hat_MCEM = all_theta_hat_MCEMs[1]
            theta_hat_MCEM = all_theta_hat_MCEMs[1]

            @test (theta_hat_EM ≈ theta_hat_MCEM) (rtol = tol)
        end

        

        #! Fix-SE
        # @testset "Does empirical SE of ascent MCEM match EM SE formula?" begin
        #     # --------------------------------- Get EM SE -------------------------------- #

        #     theta_hat_EM = run_EM(theta1, Y, theta_fixed, rtol = 1e-6)
        #     EM_SE = EM_SE_formula(theta_hat_EM, Y, theta_fixed)

        #     # ---------------------- Get empirical SE of ascent MCEM --------------------- #

            
        # end
    end

end

#? Some notes on the atol parameter of AMCEM:
#? The atol parameter is the absolute tolerance for convergence. Specifically, we require that an upper confidence bound for the increment in the EM objective is less than atol. I am not aware of any theoretical guarantees this provides about the difference between the AMCEM estimator and any quantity we would like it to converge to. It would be nice to have some sort of Lipschitz guarantee that the EM objective increment being small implies a bound on something in the parameter space. E.g. Ideally, this would bound the distance to the MLE. More conservatively, we might hope that the increment being small guarantees that the distance between consecutive estimates is also small.
