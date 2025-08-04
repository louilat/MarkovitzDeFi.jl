using MarkovitzDeFi
using Test

@testset "MarkovitzDeFi.jl" begin
    # Write your tests here.
    T = 2.
    C = [1 .5 .3; .5 1 .4; .3 .4 1]
    μ = [.1, .15, -.11]
    σ = [.30, .35, .30]
    r = [.05, .02, .03]
    lt = [.75, .80, .85]
    lb = [1.05, 1.10, 1.05]
    ω = [10, 15, -18] / 7

    model = MarkovitzDeFi.DefiMarkovitzModel(
        T, μ, r, σ, C, lt, lb, ω,
    )
    
    ν, δ, λ, γ = MarkovitzDeFi.compute_nu_delta_lambda_gamma(model; verbose = false)
    @testset "Parameters nu, ..." begin
        @test ν ≈ [0.3214, 0.6, -0.7714] atol=0.001
        @test δ ≈ .2143 atol=0.001
        @test λ ≈ .6471 atol=0.001
        @test γ ≈ .8557 atol=0.001
    end

    ψ, ϕ = MarkovitzDeFi.compute_psi_phi(model; verbose = false)
    @testset "Parameters phi and psi" begin 
        @test ψ ≈ 0.2224 atol=0.001
        @test ϕ ≈ -0.2057 atol=0.001
    end

    norm_x, x1, norm_y, y1, norm_z, z1, x_dot_y, x_plus_y_dot_z = MarkovitzDeFi.compute_x_y_z(model, ν, γ, ψ; verbose = false)
    @testset "Vectors x, y and z" begin
        @test norm_x ≈ 1.001 atol=0.001
        @test x1 ≈ .9838 atol=0.001
        @test norm_y ≈ 0.8523 atol=0.001
        @test y1 ≈ -0.8522 atol=0.001
        @test norm_z ≈ 0.2298 atol=0.001
        @test z1 ≈ -0.1316 atol=0.001
        @test x_dot_y ≈ -0.8376 atol=0.001
        @test x_plus_y_dot_z ≈ -0.0528 atol=0.001
    end

    c1, c2, c3 = MarkovitzDeFi.compute_c(model, ψ, ϕ; verbose = false)
    @testset "Parameters c" begin 
        @test c1 ≈ 1.5686 atol=0.001
        @test c2 ≈ -0.0915 atol=0.001
        @test c3 ≈ -1.5169 atol=0.001
    end

    η = λ / γ
    I0 = MarkovitzDeFi.compute_I0(η, δ, γ, T)
    I1 = MarkovitzDeFi.compute_I1(η, δ, γ, T)
    I2 = MarkovitzDeFi.compute_I2(η, δ, γ, T)
    @testset "Integrals I" begin 
        @test I0 ≈ 0.80192 atol=0.00001
        @test I1 ≈ -.13650 atol=0.00001
        @test I2 ≈ -.10672 atol=0.00001
    end

    Λ0, Λ1, Λ2, Λ3, Λ4, Λ5, Λ6 = MarkovitzDeFi.compute_all_lambdas(
        norm_x,
        norm_y,
        norm_z,
        x1,
        y1,
        z1,
        x_dot_y,
        x_plus_y_dot_z,
        λ,
        δ,
        γ,
        c1,
        c2,
        c3,
        T,
    )

    @testset "Λ parameters" begin
        @test Λ0 ≈ 4.4634 atol=0.001
        @test Λ1 ≈ -2.4496 atol=0.001
        @test Λ2 ≈ -0.0341 atol=0.001
        @test Λ3 ≈ 0.000 atol=0.001
        @test Λ4 ≈ -3.064 atol=0.001
        @test Λ5 ≈ 0.0021 atol=0.001
        @test Λ6 ≈ -0.9505 atol=0.001
    end

    M1 = MarkovitzDeFi.compute_first_order_moment_return(c1, c2, c3, η, δ, λ, γ, y1, z1, I0, I1)
    M2 = MarkovitzDeFi.compute_second_order_moment_return(η, δ, γ, Λ0, Λ1, Λ2, Λ3, Λ4, Λ5, Λ6, I0, I1, I2, T)
    Obj1 = MarkovitzDeFi.compute_objective(model, 1)
    Obj2 = MarkovitzDeFi.compute_objective(model, .7)
    @testset "First and second order moments" begin
        @test M1 ≈ 0.8229 atol=0.001
        @test M2 ≈ 2.4170 atol=0.001
        @test Obj1 ≈ 0.9169 atol=0.001
        @test Obj2 ≈ 1.1637 atol=0.001
    end


    grad1 = MarkovitzDeFi.compute_gradient(x -> sum(x.^2), [1, 3])
    grad2 = MarkovitzDeFi.compute_gradient(x -> exp(x[1]) + 2 * x[2] * x[3], [1, 2, 3])
    grad3 = MarkovitzDeFi.compute_gradient(x -> x[1]^2 * x[2], [2, 4, 1])
    @testset "Gradient computation" begin
        @test grad1 ≈ [2, 6] atol = 0.001
        @test grad2 ≈ [exp(1), 6, 4] atol = 0.001
        @test grad3 ≈ [16, 4, 0] atol = 0.001
    end

    projection1 = MarkovitzDeFi.orthogonal_projection([-.5, .5])
    projection2 = MarkovitzDeFi.orthogonal_projection([.7, .7, -1.4])
    @testset "Orthogonal projection on line" begin
        @test projection1 ≈ [-.5, .5] atol = 0.001
        @test projection2 ≈ [.7, .7, -1.4] atol = 0.001
    end

    params = MarkovitzDeFi.modelParams(T, C, μ, σ, r, lt, lb)
    objective_func = MarkovitzDeFi.wrap_objective(params, 1)
    sol1 = MarkovitzDeFi.adam_optimize(objective_func, 1500, [1, 0, 0], 5e-3, [.9, .999], 0, 1e-8; project = true)
    sol2 = MarkovitzDeFi.adam_optimize(objective_func, 1500, [0, 1, 0], 5e-3, [.9, .999], 0, 1e-8; project = true)
    sol3 = MarkovitzDeFi.adam_optimize(objective_func, 1500, [0, 0, 1], 5e-3, [.9, .999], 0, 1e-8; project = true)
    @testset "Adam Optimizer" begin
        @test sol1 ≈ [1.1466, 0.8696, -1.0162] atol = 0.001
        @test sol2 ≈ [1.1466, 0.8696, -1.0162] atol = 0.001
        @test sol3 ≈ [1.1466, 0.8696, -1.0162] atol = 0.001
    end
end
