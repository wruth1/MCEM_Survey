
@testset "Low-Level Functions" begin
    
    @testset "Counting Alleles" begin
        @testset "num_O_alleles" begin
            @test num_O_alleles([0, 0, 0, 0, 0, 0]) == 0
            @test num_O_alleles([1, 0, 0, 0, 0, 0]) == 2
            @test num_O_alleles([0, 1, 0, 0, 0, 0]) == 1
            @test num_O_alleles([0, 0, 1, 0, 0, 0]) == 0
            @test num_O_alleles([0, 0, 0, 1, 0, 0]) == 1
            @test num_O_alleles([0, 0, 0, 0, 1, 0]) == 0
            @test num_O_alleles([0, 0, 0, 0, 0, 1]) == 0
            @test num_O_alleles([1, 1, 1, 1, 1, 1]) == 4
        end

        @testset "num_A_alleles" begin
            @test num_A_alleles([0, 0, 0, 0, 0, 0]) == 0
            @test num_A_alleles([1, 0, 0, 0, 0, 0]) == 0
            @test num_A_alleles([0, 1, 0, 0, 0, 0]) == 1
            @test num_A_alleles([0, 0, 1, 0, 0, 0]) == 2
            @test num_A_alleles([0, 0, 0, 1, 0, 0]) == 0
            @test num_A_alleles([0, 0, 0, 0, 1, 0]) == 0
            @test num_A_alleles([0, 0, 0, 0, 0, 1]) == 1
            @test num_A_alleles([1, 1, 1, 1, 1, 1]) == 4
        end

        @testset "num_B_alleles" begin
            @test num_B_alleles([0, 0, 0, 0, 0, 0]) == 0
            @test num_B_alleles([1, 0, 0, 0, 0, 0]) == 0
            @test num_B_alleles([0, 1, 0, 0, 0, 0]) == 0
            @test num_B_alleles([0, 0, 1, 0, 0, 0]) == 0
            @test num_B_alleles([0, 0, 0, 1, 0, 0]) == 1
            @test num_B_alleles([0, 0, 0, 0, 1, 0]) == 2
            @test num_B_alleles([0, 0, 0, 0, 0, 1]) == 1
            @test num_B_alleles([1, 1, 1, 1, 1, 1]) == 4
        end

        @testset "Total number of alleles should be double the sample size" begin
            @test sum(num_OAB_alleles(X)) == 2 * sum(X)
        end
    end
end