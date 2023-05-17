
#* This comment color denotes when something needs to be changed before running test on a more complicated problem.


@testset "MCEM Only" begin
    # Note: The tolerance value for these tests must be calibrated for different distributions
    @testset "MCEM Algorithm" begin
        @testset "Complete data likelihood functions" begin

            @testset "Is complete data score equal to gradient of complete data log-lik?" begin
                # Function to differentiate numerically
                function this_log_lik(theta)
                    return complete_data_log_lik(theta, Y, X)
                    # return fun(theta, Y, X)
                end


                # ---------------------- Compare at true value of theta --------------------- #
                # Compute score
                score0 = complete_data_score(theta0, Y, X)

                # Compute gradient of log-lik
                grad_log_lik0 = ReverseDiff.gradient(this_log_lik, theta0)
                
                @test score0 ≈ grad_log_lik0


                # ------------------ Compare away from true value of theta ------------------ #
                # Compute score
                score1 = complete_data_score(theta1, Y, X)

                # Compute gradient of log-lik
                grad_log_lik1 = ReverseDiff.gradient(this_log_lik, theta1)

                @test score1 ≈ grad_log_lik1
            end
        end

        @testset "Conditional Sampler" begin
            # Note: We reset the seed immediately before every simulation so that order of tests doesn't influence results

            # Number of times to sample from cond'l distribution
            M = 10000

            # Relative tolerance for various moments
            mean_tol = 1e-5
            var_tol = 0.05
            mu2_tol = 1e-5


            # Generate conditional samples
            Random.seed!(1)
            all_Xs0 = sample_X_given_Y_iid(M, theta0, Y)   # theta equal to true value
            Random.seed!(1)
            all_Xs1 = sample_X_given_Y_iid(M, theta1, Y)   # theta different from true value


            @testset "Does conditional sampler get the moments right?" begin
                @testset "Conditional mean" begin
                    # ---------------------------- Starting at theta0 ---------------------------- #
                    
                    # Compute mean of Xs
                    mean_X0 = mean(all_Xs0)

                    # Analytical mean
                    true_mean_X0 = mu_X_given_Y(theta0, Y)

                    # norm(mean_X0 - true_mean_X0) / max(norm(mean_X0), norm(true_mean_X0))
                    @test isapprox(mean_X0, true_mean_X0, rtol = mean_tol)


                    # ---------------------------- Starting at theta1 ---------------------------- #
                    
                    # Compute mean of Xs
                    mean_X1 = mean(all_Xs1)

                    # Analytical mean
                    true_mean_X1 = mu_X_given_Y(theta1, Y)

                    # norm(mean_X1 - true_mean_X1) / max(norm(mean_X1), norm(true_mean_X1))
                    @test isapprox(mean_X1, true_mean_X1, rtol = mean_tol)
                end

                @testset "Conditional variance" begin
                    # ---------------------------- Starting at theta0 ---------------------------- #
                    cov_X0 = cov(all_Xs0)

                    # # Evaluate correlations based on cov_X0
                    # cor_X0 = deepcopy(cov_X0)
                    # for i in 2:5, j in 2:5
                    #     cor_X0[i, j] = cov_X0[i, j] / sqrt(cov_X0[i, i] * cov_X0[j, j])
                    # end
                    # cor_X0

                    # Analytical variance
                    true_cov_X0 = cov_X_given_Y(theta0, Y)

                    # # Analytical correlation matrix
                    # true_cor_X0 = deepcopy(true_cov_X0)
                    # for i in 2:5, j in 2:5
                    #     true_cor_X0[i, j] = true_cov_X0[i, j] / sqrt(true_cov_X0[i, i] * true_cov_X0[j, j])
                    # end


                    # norm(cov_X0 - true_cov_X0) / max(norm(cov_X0), norm(true_cov_X0))
                    @test isapprox(cov_X0, true_cov_X0, rtol = var_tol)

                    # norm(cor_X0 - true_cor_X0) / max(norm(cor_X0), norm(true_cor_X0))


                    # ---------------------------- Starting at theta1 ---------------------------- #
                    
                    ### Pooled estimate of conditional variance of X
                    # Estimate variance of each X component separately
                    cov_X1 = cov(all_Xs1)

                    # Analytical variance
                    true_var_X1 = var_X_given_Y(theta1)

                    @test isapprox(var_X1, true_var_X1, rtol = var_tol)
                end

                @testset "Conditional second moment" begin
                    all_X0s_sq = [X * X' for X in all_Xs0]
                    mean_X0_sq = mean(all_X0s_sq)

                    true_mean_X0_sq = mu2_X_given_Y(theta0, Y)

                    # norm(mean_X0_sq - true_mean_X0_sq) / max(norm(mean_X0_sq), norm(true_mean_X0_sq))
                    @test isapprox(mean_X0_sq, true_mean_X0_sq, rtol = mu2_tol)



                    all_X1s_sq = [X * X' for X in all_Xs1]
                    mean_X1_sq = mean(all_X1s_sq)

                    true_mean_X1_sq = mu2_X_given_Y(theta1, Y)

                    # norm(mean_X1_sq - true_mean_X1_sq) / max(norm(mean_X1_sq), norm(true_mean_X1_sq))
                    @test isapprox(mean_X1_sq, true_mean_X1_sq, rtol = mu2_tol)
                end
            end
        end


        @testset "MCEM Objective Function" begin
            
            @testset "Does difference of MCEM objective match MCEM increment?" begin
                M = 1000

                # ---------------------- Starting at true value of theta --------------------- #

                Random.seed!(1)
                theta_hat0, all_Xs0 = MCEM_update_iid(theta0, Y, M, return_X = true)

                Q_old = Q_MCEM_iid(theta0, Y, all_Xs0)
                Q_new = Q_MCEM_iid(theta_hat0, Y, all_Xs0)
                Q_diff0 = Q_new - Q_old

                Q_increment0 = Q_MCEM_increment_iid(theta_hat0, theta0, Y, all_Xs0)

                @test Q_diff0 ≈ Q_increment0


                # ------------------ Starting away from true value of theta ------------------ #
                Random.seed!(1)
                theta_hat1, all_Xs1 = MCEM_update_iid(theta1, Y, M, return_X = true)

                Q_old = Q_MCEM_iid(theta1, Y, all_Xs1)
                Q_new = Q_MCEM_iid(theta_hat1, Y, all_Xs1)
                Q_diff1 = Q_new - Q_old

                Q_increment1 = Q_MCEM_increment_iid(theta_hat1, theta1, Y, all_Xs1)

                @test Q_diff1 ≈ Q_increment1
            end

        end


        @testset "Maximizer of MCEM objective" begin
            
            @testset "Does MCEM estimate maximize the MCEM objective?" begin
                M = 1000
                
                Random.seed!(1)
                noise_sizes = randn(100)
                noise_grid = [0.1 * theta0 .* noise_sizes[i] for i in eachindex(noise_sizes)]

                # ---------------------------- Starting at theta0 ---------------------------- #
                theta_old = theta0

                Random.seed!(1)
                theta_hat0, all_Xs0 = MCEM_update_iid(theta0, Y, M, return_X = true)
                Q_max = Q_MCEM_iid(theta_hat0, Y, all_Xs0)

                nearby_theta_values = [theta_hat0 + noise_grid[i] for i in eachindex(noise_grid)]

                nearby_Q_values = [Q_MCEM_iid(theta, Y, all_Xs0) for theta in nearby_theta_values]

                @test all(Q_max .>= nearby_Q_values)

                # ---------------------------- Starting at theta1 ---------------------------- #

                theta_hat1, all_Xs1 = MCEM_update_iid(theta1, Y, M, return_X = true)
                Q_max = Q_MCEM_iid(theta_hat1, Y, all_Xs1)

                nearby_theta_values = [theta_hat1 + noise_grid[i] for i in eachindex(noise_grid)]

                nearby_Q_values = [Q_MCEM_iid(theta, Y, all_Xs1) for theta in nearby_theta_values]

                @test all(Q_max .>= nearby_Q_values)            
            end

            
        end
    end





    @testset "Ascent-Based MCEM" begin
            
        #* Adjust tolerances for new distributions
        @testset "Is asymptotic standard error accurate?" begin
            
            easy_tol = 0.05
            hard_tol = 0.01

            # Number of Monte Carlo samples to draw
            M = 1000

            # Number of times to replicate iteration
            B = 100

            # ---------------------------------------------------------------------------- #
            #                      Starting at the true value of theta                     #
            # ---------------------------------------------------------------------------- #

            Random.seed!(1)

            all_ASEs = zeros(B)
            all_increments = zeros(B)


            for b in 1:B
            # @showprogress for b in 1:B

                # Generate conditional sample
                all_Xs = sample_X_given_Y(theta0, Y, M)

                # Run MCEM
                theta_hat = MCEM_update(Y, all_Xs)

                # Estimated ASE
                all_ASEs[b] = get_ASE(theta_hat, theta0, Y, all_Xs)

                # Improvement in MCEM objective function
                all_increments[b] = Q_MCEM_increment(theta_hat, theta0, Y, all_Xs)
            end

            # Average ASE
            mean_ASE = mean(all_ASEs)

            # Empirical SE of MCEM increment
            empirical_SE = std(all_increments)

            @test (mean_ASE ≈ empirical_SE) (rtol = hard_tol)


            # ---------------------------------------------------------------------------- #
            #                    Starting away from true value of theta                    #
            # ---------------------------------------------------------------------------- #

            Random.seed!(1)

            all_ASEs = zeros(B)
            all_increments = zeros(B)


            for b in 1:B
            # @showprogress for b in 1:B

                # Generate conditional sample
                all_Xs = sample_X_given_Y(theta1, Y, M)

                # Run MCEM
                theta_hat = MCEM_update(Y, all_Xs)

                # Estimated ASE
                all_ASEs[b] = get_ASE(theta_hat, theta1, Y, all_Xs)

                # Improvement in MCEM objective function
                all_increments[b] = Q_MCEM_increment(theta_hat, theta1, Y, all_Xs)
            end

            # Average ASE
            mean_ASE = mean(all_ASEs)

            # Empirical SE of MCEM increment
            empirical_SE = std(all_increments)

            @test (mean_ASE ≈ empirical_SE) (rtol = easy_tol)
        end


        


        @testset "Is ascent MCEM update close to ordinary MCEM update?" begin

            # Number of Monte Carlo samples to draw
            M = 1000

            # Relative tolerance
            tol = 0.01

            # Ordinary MCEM updates
            Random.seed!(1)
            theta_hat_MCEM0 = MCEM_update(theta0, Y, M)
            Random.seed!(1)
            theta_hat_MCEM1 = MCEM_update(theta1, Y, M)

            # Ascent MCEM updates
            Random.seed!(1)
            theta_hat_ascent_MCEM0 = ascent_MCEM_update(theta0, Y, M, 0.05, 3)
            Random.seed!(1)
            theta_hat_ascent_MCEM1 = ascent_MCEM_update(theta1, Y, M, 0.05, 3)

            @test isapprox(theta_hat_ascent_MCEM0, theta_hat_MCEM0, rtol = tol)
            @test isapprox(theta_hat_ascent_MCEM1, theta_hat_MCEM1, rtol = tol)
        end


        # @testset "Do empirical and estimated SEs match for ascent MCEM?" begin
        #     #! Standard error formula still isn't working correctly
        #     #! Fix-SE

        #     # Relative tolerance for comparing AMCEM SEs
        #     # Generous due to limited number of samples
        #     rtol = 0.1

        #     empirical_SE = std(all_theta_hat_MCEMs)

        #     mean_SE_hat = mean(all_SE_hat_MCEMs)

        #     @test (empirical_SE ≈ mean_SE_hat) (rtol = rtol)
        # end
        
    end



end



#? Some notes on the atol parameter of AMCEM:
#? The atol parameter is the absolute tolerance for convergence. Specifically, we require that an upper confidence bound for the increment in the EM objective is less than atol. I am not aware of any theoretical guarantees this provides about the difference between the AMCEM estimator and any quantity we would like it to converge to. It would be nice to have some sort of Lipschitz guarantee that the EM objective increment being small implies a bound on something in the parameter space. E.g. Ideally, this would bound the distance to the MLE. More conservatively, we might hope that the increment being small guarantees that the distance between consecutive estimates is also small.