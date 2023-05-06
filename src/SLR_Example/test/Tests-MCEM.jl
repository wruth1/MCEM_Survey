
#* This comment color denotes when something needs to be changed before running test on a more complicated problem.


@testset "MCEM Only" begin
    # Note: The tolerance value for these tests must be calibrated for different distributions
    @testset "MCEM Algorithm" begin
        @testset "Complete data likelihood functions" begin

            @testset "Is complete data score equal to gradient of complete data log-lik?" begin
                # Function to differentiate numerically
                function this_log_lik(theta)
                    return complete_data_log_lik(theta, Y, X, theta_fixed)
                    # return fun(theta, Y, X, theta_fixed)
                end


                # ---------------------- Compare at true value of theta --------------------- #
                # Compute score
                score1 = complete_data_score(theta1, Y, X, theta_fixed)

                # Compute gradient of log-lik
                grad_log_lik1 = ReverseDiff.gradient(this_log_lik, theta1)
                
                @test score1 ≈ grad_log_lik1


                # ------------------ Compare away from true value of theta ------------------ #
                # Compute score
                score2 = complete_data_score(theta2, Y, X, theta_fixed)

                # Compute gradient of log-lik
                grad_log_lik2 = ReverseDiff.gradient(this_log_lik, theta2)

                @test score2 ≈ grad_log_lik2
            end
        end

        @testset "Conditional Sampler" begin
            # Note: We reset the seed immediately before every simulation so that order of tests doesn't influence results

            # Number of times to sample from cond'l distribution
            M = 10000

            # Relative tolerance for various moments
            mean_tol = 0.01
            var_tol = 0.01
            mu2_tol = 0.01
            mu3_tol = 0.05
            mu4_tol = 0.05

            # Generate conditional samples
            Random.seed!(1)
            all_Xs1 = sample_X_given_Y(theta1, Y, theta_fixed, M)   # theta equal to true value
            Random.seed!(1)
            all_Xs2 = sample_X_given_Y(theta2, Y, theta_fixed, M)   # theta different from true value


            @testset "Does conditional sampler get the moments right?" begin
                @testset "Conditional mean" begin
                    # ---------------------------- Starting at theta1 ---------------------------- #
                    
                    # Compute mean of Xs
                    mean_X1 = mean(all_Xs1)

                    # Analytical mean
                    true_mean_X1 = [mu_X_given_Y(theta1, y, theta_fixed) for y in Y]

                    @test isapprox(mean_X1, true_mean_X1, rtol = mean_tol)


                    # ---------------------------- Starting at theta2 ---------------------------- #
                    
                    # Compute mean of Xs
                    mean_X2 = mean(all_Xs2)

                    # Analytical mean
                    true_mean_X2 = [mu_X_given_Y(theta2, y, theta_fixed) for y in Y]

                    @test isapprox(mean_X2, true_mean_X2, rtol = mean_tol)
                end

                @testset "Conditional variance" begin
                    # ---------------------------- Starting at theta1 ---------------------------- #
                    
                    ### Pooled estimate of conditional variance of X
                    # Estimate variance of each X component separately
                    all_X_components1 = [[all_Xs1[i][j] for i in 1:M] for j in eachindex(Y)]
                    all_X_vars1 = Statistics.var.(all_X_components1)
                    var_X1 = mean(all_X_vars1)

                    # Analytical variance
                    true_var_X1 = var_X_given_Y(theta1, theta_fixed)

                    @test isapprox(var_X1, true_var_X1, rtol = var_tol)


                    # ---------------------------- Starting at theta2 ---------------------------- #
                    
                    ### Pooled estimate of conditional variance of X
                    # Estimate variance of each X component separately
                    all_X_components2 = [[all_Xs2[i][j] for i in 1:M] for j in eachindex(Y)]
                    all_X_vars2 = Statistics.var.(all_X_components2)
                    var_X2 = mean(all_X_vars2)

                    # Analytical variance
                    true_var_X2 = var_X_given_Y(theta2, theta_fixed)

                    @test isapprox(var_X2, true_var_X2, rtol = var_tol)
                end

                @testset "Conditional second moment" begin
                    all_X_components1 = [[all_Xs1[i][j] for i in 1:M] for j in eachindex(Y)]
                    all_mu2s1 = [mean(X.^2) for X in all_X_components1]

                    true_mu2s1 = [mu2_X_given_Y(theta1, y, theta_fixed) for y in Y]

                    @test isapprox(all_mu2s1, true_mu2s1, rtol = mu2_tol)



                    all_X_components2 = [[all_Xs2[i][j] for i in 1:M] for j in eachindex(Y)]
                    all_mu2s2 = [mean(X.^2) for X in all_X_components2]

                    true_mu2s2 = [mu2_X_given_Y(theta2, y, theta_fixed) for y in Y]

                    @test isapprox(all_mu2s2, true_mu2s2, rtol = mu2_tol)
                end


                @testset "Conditional third moment" begin
                    all_X_components1 = [[all_Xs1[i][j] for i in 1:M] for j in eachindex(Y)]
                    all_mu3s1 = [mean(X.^3) for X in all_X_components1]

                    true_mu3s1 = [mu3_X_given_Y(theta1, y, theta_fixed) for y in Y]

                    @test isapprox(all_mu3s1, true_mu3s1, rtol = mu3_tol)



                    all_X_components2 = [[all_Xs2[i][j] for i in 1:M] for j in eachindex(Y)]
                    all_mu3s2 = [mean(X.^3) for X in all_X_components2]

                    true_mu3s2 = [mu3_X_given_Y(theta2, y, theta_fixed) for y in Y]

                    @test isapprox(all_mu3s2, true_mu3s2, rtol = mu3_tol)
                end



                @testset "Conditional fourth moment" begin
                    all_X_components1 = [[all_Xs1[i][j] for i in 1:M] for j in eachindex(Y)]
                    all_mu4s1 = [mean(X.^4) for X in all_X_components1]

                    true_mu4s1 = [mu4_X_given_Y(theta1, y, theta_fixed) for y in Y]

                    @test isapprox(all_mu4s1, true_mu4s1, rtol = mu4_tol)



                    all_X_components2 = [[all_Xs2[i][j] for i in 1:M] for j in eachindex(Y)]
                    all_mu4s2 = [mean(X.^4) for X in all_X_components2]

                    true_mu4s2 = [mu4_X_given_Y(theta2, y, theta_fixed) for y in Y]

                    @test isapprox(all_mu4s2, true_mu4s2, rtol = mu4_tol)
                end
            end
        end


        @testset "MCEM Objective Function" begin
            
            @testset "Does difference of MCEM objective match MCEM increment?" begin
                M = 1000

                # ---------------------- Starting at true value of theta --------------------- #

                Random.seed!(1)
                theta_hat1, all_Xs1 = MCEM_update(theta1, Y, theta_fixed, M, return_X = true)

                Q_old = Q_MCEM(theta1, Y, all_Xs1, theta_fixed)
                Q_new = Q_MCEM(theta_hat1, Y, all_Xs1, theta_fixed)
                Q_diff1 = Q_new - Q_old

                Q_increment1 = Q_MCEM_increment(theta_hat1, theta1, Y, all_Xs1, theta_fixed)

                @test Q_diff1 ≈ Q_increment1


                # ------------------ Starting away from true value of theta ------------------ #
                Random.seed!(1)
                theta_hat2, all_Xs2 = MCEM_update(theta2, Y, theta_fixed, M, return_X = true)

                Q_old = Q_MCEM(theta2, Y, all_Xs2, theta_fixed)
                Q_new = Q_MCEM(theta_hat2, Y, all_Xs2, theta_fixed)
                Q_diff2 = Q_new - Q_old

                Q_increment2 = Q_MCEM_increment(theta_hat2, theta2, Y, all_Xs2, theta_fixed)

                @test Q_diff2 ≈ Q_increment2
            end

        end


        @testset "Maximizer of MCEM objective" begin
            
            @testset "Does MCEM estimate maximize the MCEM objective?" begin
                M = 1000
                
                Random.seed!(1)
                noise_sizes = randn(100)
                noise_grid = [0.1 * theta1 .* noise_sizes[i] for i in eachindex(noise_sizes)]

                # ---------------------------- Starting at theta1 ---------------------------- #
                theta_old = theta1

                Random.seed!(1)
                theta_hat1, all_Xs1 = MCEM_update(theta1, Y, theta_fixed, M, return_X = true)
                Q_max = Q_MCEM(theta_hat1, Y, all_Xs1, theta_fixed)

                nearby_theta_values = [theta_hat1 + noise_grid[i] for i in eachindex(noise_grid)]

                nearby_Q_values = [Q_MCEM(theta, Y, all_Xs1, theta_fixed) for theta in nearby_theta_values]

                @test all(Q_max .>= nearby_Q_values)

                # ---------------------------- Starting at theta2 ---------------------------- #

                theta_hat2, all_Xs2 = MCEM_update(theta2, Y, theta_fixed, M, return_X = true)
                Q_max = Q_MCEM(theta_hat2, Y, all_Xs2, theta_fixed)

                nearby_theta_values = [theta_hat2 + noise_grid[i] for i in eachindex(noise_grid)]

                nearby_Q_values = [Q_MCEM(theta, Y, all_Xs2, theta_fixed) for theta in nearby_theta_values]

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
                all_Xs = sample_X_given_Y(theta1, Y, theta_fixed, M)

                # Run MCEM
                theta_hat = MCEM_update(Y, all_Xs)

                # Estimated ASE
                all_ASEs[b] = get_ASE(theta_hat, theta1, Y, all_Xs, theta_fixed)

                # Improvement in MCEM objective function
                all_increments[b] = Q_MCEM_increment(theta_hat, theta1, Y, all_Xs, theta_fixed)
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
                all_Xs = sample_X_given_Y(theta2, Y, theta_fixed, M)

                # Run MCEM
                theta_hat = MCEM_update(Y, all_Xs)

                # Estimated ASE
                all_ASEs[b] = get_ASE(theta_hat, theta2, Y, all_Xs, theta_fixed)

                # Improvement in MCEM objective function
                all_increments[b] = Q_MCEM_increment(theta_hat, theta2, Y, all_Xs, theta_fixed)
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
            theta_hat_MCEM1 = MCEM_update(theta1, Y, theta_fixed, M)
            Random.seed!(1)
            theta_hat_MCEM2 = MCEM_update(theta2, Y, theta_fixed, M)

            # Ascent MCEM updates
            Random.seed!(1)
            theta_hat_ascent_MCEM1 = ascent_MCEM_update(theta1, Y, theta_fixed, M, 0.05, 3)
            Random.seed!(1)
            theta_hat_ascent_MCEM2 = ascent_MCEM_update(theta2, Y, theta_fixed, M, 0.05, 3)

            @test isapprox(theta_hat_ascent_MCEM1, theta_hat_MCEM1, rtol = tol)
            @test isapprox(theta_hat_ascent_MCEM2, theta_hat_MCEM2, rtol = tol)
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