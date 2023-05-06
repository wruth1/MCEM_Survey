
#* This comment color denotes when something needs to be changed before running test on a more complicated problem.

@testset "Obs data lik, EM and MCEM" begin

    @testset "Observed Data Likelihood Functions" begin
        @testset "Does analytical score match numerical approximation?" begin

            # ---------------------- Compare at true value of theta ---------------------- #
            
            score = obs_data_score(theta1, Y)

            function this_log_lik(this_theta)
                return obs_data_log_lik(this_theta, Y)
            end
            approx_score = ReverseDiff.gradient(this_log_lik, theta1)
        
            # Test
            @test score ≈ approx_score


        # ------------------- Compare away from true value of theta ------------------ #
            score2 = obs_data_score(theta2, Y)
            approx_score2 = ReverseDiff.gradient(this_log_lik, theta2)
            @test score2 ≈ approx_score2
        end

        @testset "Does analytical Hessian match numerical approximation?" begin
            hessian = obs_data_Hessian(theta1, Y)

            function this_log_lik(this_theta)
                return obs_data_log_lik(this_theta, Y)
            end
            approx_hessian = ReverseDiff.hessian(this_log_lik, theta1)

            @test hessian ≈ approx_hessian
        end

        @testset "Is Hessian negative definite at the MLE?" begin
            theta_hat = obs_data_MLE(Y)
            hessian = obs_data_Hessian(theta_hat, Y)

            if size(hessian, 1) == 1
                @test hessian[1] < 0
            else
                @test all(eigvals(hessian) .< 0)
            end
        end

        @testset "Does MLE maximize the likelihood?" begin
            theta_hat = obs_data_MLE(Y)
            max_lik_val = obs_data_log_lik(theta_hat, Y)
            
            Random.seed!(1)
            noise_sizes = randn(100)
            noise_grid = [0.1 * theta_hat .* noise_sizes[i] for i in eachindex(noise_sizes)]
    
            nearby_theta_values = [theta_hat + noise_grid[i] for i in eachindex(noise_grid)]
            nearby_lik_vals = [obs_data_log_lik(theta, Y) for theta in nearby_theta_values]
            
            @test all(max_lik_val .>= nearby_lik_vals)
        end


        #! START HERE

        @testset "Is MLE standard error formula accurate?" begin
            # Increase sample size to invoke asymptotic SE formula
            this_n = 100

            # Number of datasets to simulate
            B = 10000

            # Tolerance for comparing empirical and analytical SEs
            # Value is pretty generous. Getting better precision takes uncomfortably more computing time.
            rtol = 0.05

            all_MLEs = Vector{Any}(undef, B)
            all_SEs = Vector{Any}(undef, B)

            Random.seed!(1)
            for b in 1:B
                # Generate data
                X = rand(Normal(mu_0, tau_0), this_n)
                epsilon = rand(Normal(0, sigma_0), this_n)
                Y = beta_0 * X + epsilon

                # Compute MLE and SE
                theta_hat = obs_data_MLE(Y)
                all_MLEs[b] = theta_hat
                all_SEs[b] = obs_data_MLE_SE(theta_hat, Y)
            end

            empirical_SE = std(all_MLEs)
            analytical_SE = mean(all_SEs)

            @test (empirical_SE ≈ analytical_SE) (rtol = rtol)
        end

    end



    @testset "Complete Data Likelihood Functions" begin

        @testset "Score" begin

            @testset "Does score match numerical gradient of log-likelihood?" begin

                # ---------------------- Compare at true value of theta ---------------------- #
                score = complete_data_score(theta1, Y, X)

                function this_log_lik(this_theta)
                    return complete_data_log_lik(this_theta, Y, X)
                end
                approx_score = ReverseDiff.gradient(this_log_lik, theta1)
            
                # Test
                @test score ≈ approx_score

                # Another value of theta
                score2 = complete_data_score(theta2, Y, X)
                approx_score2 = ReverseDiff.gradient(this_log_lik, theta2)
                @test score2 ≈ approx_score2
            end
        end

        @testset "Hessian" begin
            @testset "Is Hessian equal to second derivative of log-likelihood?" begin
                hessian = complete_data_Hessian(theta, Y, X)

                function this_log_lik(this_theta)
                    return complete_data_log_lik(this_theta, Y, X)
                end
                approx_hessian = ReverseDiff.hessian(this_log_lik, theta)

                @test hessian ≈ approx_hessian
            end

        end
    end

        



    @testset "EM Algorithm" begin
        @testset "Does EM improve the obs data likelihood?" begin
            convergence_tol = 1e-6

            # ---------------------------- Starting at theta1 ---------------------------- #
            _, theta_hat_trajectory1 = run_EM(theta1, Y, rtol = convergence_tol, return_trajectory = true)
            all_obs_log_liks1 = [obs_data_log_lik(theta, Y) for theta in theta_hat_trajectory1]
            all_improvements1 = diff(all_obs_log_liks1)

            @test all(all_improvements1 .>= 0)


            # ---------------------------- Starting at theta2 ---------------------------- #
            _, theta_hat_trajectory2 = run_EM(theta2, Y, rtol = convergence_tol, return_trajectory = true)
            all_obs_log_liks2 = [obs_data_log_lik(theta, Y) for theta in theta_hat_trajectory2]
            all_improvements2 = diff(all_obs_log_liks2)

            @test all(all_improvements2 .>= 0)
        end

        @testset "Does EM converge to the MLE?" begin
            #* Note: This test will need to be modified if the obs data likelihood has multiple modes
            convergence_tol = 1e-6
            estimation_tol = 1e-3

            # Directly maximize the observed data likelihood
            theta_hat_MLE = obs_data_MLE(Y)

            # -------------------- Starting at the true value of theta ------------------- #
            theta_hat_EM1 = run_EM(theta1, Y, rtol = convergence_tol)
            @test (theta_hat_EM1 ≈ theta_hat_MLE) (rtol = estimation_tol)

            # -------------------- Starting at a wrong value of theta -------------------- #
            theta_hat_EM2 = run_EM(theta2, Y, rtol = convergence_tol)
            @test (theta_hat_EM2 ≈ theta_hat_MLE) (rtol = estimation_tol)
        end
        
        # @testset "Do covariance matrices from EM and obs data likelihood match?" begin
            #! Fix-SE

        #     #todo: Use the conditional sampler for X to investigate the individual components of the EM covariance matrix.
            
            

            # # Tolerance for assessing convergence of EM algorithm
            # conv_tol = 1e-8
            # # Tolerance for comparing SEs from EM and MLE
            # rtol = 1e-4

            # # --------------------- Starting from true value of theta -------------------- #
            # theta_hat1 = run_EM(theta1, Y, rtol=conv_tol)

            # COV_EM = EM_COV_formula(theta_hat1, Y)

            # theta_hat_MLE = obs_data_MLE(Y)

            # COV_MLE = obs_data_MLE_Cov(theta_hat_MLE, Y)

            # # EM_COV_formula(run_EM(theta1, Y, rtol=1e-10), Y)

            # @test (SE_EM ≈ SE_MLE) (rtol = rtol)
        # end
    end




    @testset "Ascent-Based MCEM Algorithm" begin
        
        @testset "Does ascent MCEM (approximately) converge to the MLE?" begin

            # Relative tolerance for comparing MLE and ascent MCEM estimates
            tol = 5e-2

            # Maximizer of the observed data likelihood
            theta_hat_MLE = obs_data_MLE(Y)

            # --------------------------- Ascent MCEM estimator -------------------------- #
            theta_hat_MCEM = all_theta_hat_MCEMs[1]

            @test (theta_hat_MLE ≈ theta_hat_MCEM) (rtol = tol)

        end
    end
end

#? Some notes on the atol parameter of AMCEM:
#? The atol parameter is the absolute tolerance for convergence. Specifically, we require that an upper confidence bound for the increment in the EM objective is less than atol. I am not aware of any theoretical guarantees this provides about the difference between the AMCEM estimator and any quantity we would like it to converge to. It would be nice to have some sort of Lipschitz guarantee that the EM objective increment being small implies a bound on something in the parameter space. E.g. Ideally, this would bound the distance to the MLE. More conservatively, we might hope that the increment being small guarantees that the distance between consecutive estimates is also small.
