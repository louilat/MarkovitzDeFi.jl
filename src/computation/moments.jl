include("parameters.jl")
include("integrals.jl")

function compute_first_order_moment_return(
    c1::Float64,
    c2::Float64,
    c3::Float64,
    η::Float64,
    δ::Float64,
    λ::Float64,
    γ::Float64,
    y1::Float64,
    z1::Float64,
    I0::Float64,
    I1::Float64,
)
    coeff = exp(-η * δ / γ)
    term0 = c3 - δ / γ * (y1 + z1)
    term1 = y1 + 1 / η * (z1 * λ / γ - c2)
    return c1 + coeff * (term0 * I0 + term1 * I1)
end

function compute_second_order_moment_return(
    η::Float64,
    δ::Float64,
    γ::Float64,
    Λ0::Float64,
    Λ1::Float64,
    Λ2::Float64,
    Λ3::Float64,
    Λ4::Float64,
    Λ5::Float64,
    Λ6::Float64,
    I0::Float64,
    I1::Float64,
    I2::Float64,
    T::Float64,
)::Float64
    coeff = exp(- η * δ / γ)
    term0 = Λ1 - δ / γ * Λ4 + (T + (δ/γ)^2) * Λ6
    term1 = 1/η * (δ/γ + 1/η) * Λ5 - 1/η * Λ2 - 1/η^3 * Λ3 - 2 * δ/γ * Λ6 + Λ4
    term2 = 1/η^2 * Λ3 + Λ6 - 1/η * Λ5
    return Λ0 + coeff * (term0 * I0 + term1 * I1 + term2 * I2)
end

function compute_objective(model::DefiMarkovitzModel, Φ::Real; verbose = false)::Real
    ν, δ, λ, γ = compute_nu_delta_lambda_gamma(model; verbose = verbose)
    if δ <= 0 || γ <= 0 || λ == 0
        return NaN
    end
    ψ, φ = compute_psi_phi(model; verbose = verbose)
    norm_x, x1, norm_y, y1, norm_z, z1, x_dot_y, x_plus_y_dot_z = compute_x_y_z(model, ν, γ, ψ; verbose = verbose)
    c1, c2, c3 = compute_c(model, ψ, φ; verbose = verbose)
    Λ0, Λ1, Λ2, Λ3, Λ4, Λ5, Λ6 = compute_all_lambdas(
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
        model.horizon,
    )

    η = λ / γ
    I0 = compute_I0(η, δ, γ, model.horizon)
    I1 = compute_I1(η, δ, γ, model.horizon)
    I2 = compute_I2(η, δ, γ, model.horizon)
    M1 = compute_first_order_moment_return(c1, c2, c3, η, δ, λ, γ, y1, z1, I0, I1)
    M2 = compute_second_order_moment_return(
        η,
        δ,
        γ,
        Λ0,
        Λ1,
        Λ2,
        Λ3,
        Λ4,
        Λ5,
        Λ6,
        I0,
        I1,
        I2,
        model.horizon,
    )
    return M2 - M1^2 - Φ * M1
end