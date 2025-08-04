include("model.jl")

function compute_x_y_z(
    model::DefiMarkovitzModel, nu::Vector{Float64}, gamma::Float64, psi::Float64; verbose = true
)::Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64}
    K = diagm(model.sigmas) * model.correlations * diagm(model.sigmas)
    norm_x = transpose(model.weights) * K * model.weights
    norm_y = transpose(psi .* model.longs - model.weights) * K * (psi .* model.longs - model.weights)
    norm_z = psi^2 * transpose(model.longs) * K * model.longs
    norm_x = sqrt(norm_x)
    norm_y = sqrt(norm_y)
    norm_z = sqrt(norm_z)

    x1 = transpose(nu) * model.correlations * diagm(model.sigmas) * model.weights / gamma
    y1 = transpose(nu) * model.correlations * diagm(model.sigmas) * (psi .* model.longs - model.weights) / gamma
    z1 = - psi * (transpose(nu) * model.correlations * diagm(model.sigmas) * model.longs / gamma)

    x_dot_y = transpose(model.weights) * K * (psi * model.longs .- model.weights)
    x_plus_y_dot_z = -psi^2 * transpose(model.longs) * K * model.longs

    if verbose
        @info "Successfully computed x and y\n   --> norm_x = $norm_x\n   --> x1 = $x1\n   --> norm_y = $norm_y\n   --> y1 = $y1\n   --> (x|y) = $x_dot_y"
    end
    return norm_x, x1, norm_y, y1, norm_z, z1, x_dot_y, x_plus_y_dot_z
end

function compute_c(model::DefiMarkovitzModel, psi::Float64, phi::Float64; verbose = true)::Tuple{Float64, Float64, Float64}
    c1 = transpose(model.weights) * (model.mus .+ model.apy) * model.horizon
    c2 = transpose(model.weights) * model.apy - psi * transpose(model.longs) * (model.mus .+ model.apy)
    c3 = transpose(psi .* model.longs .- model.weights) * (model.mus .+ model.apy) * model.horizon + phi
    if verbose
        @info "Successfully computed c\n   --> c1 = $c1\n   --> c2 = $c2\n   --> c3 = $c3"
    end
    return c1, c2, c3
end

function compute_lambda_0(c1::Float64, norm_x::Float64, horizon::Float64)
    return c1^2 + norm_x^2 * horizon
end

function compute_lambda_1(
    norm_y::Float64,
    x1::Float64,
    y1::Float64,
    z1::Float64,
    x_dot_y::Float64,
    delta::Float64,
    gamma::Float64,
    c1::Float64,
    c3::Float64,
    horizon::Float64
)::Float64
    term1 = (norm_y^2 - y1^2 + 2 * x_dot_y - 2 * x1 * y1) * horizon
    term2 = z1 * delta / gamma * (z1 * delta / gamma - 2 * (c1 + c3))
    term3 = c3 * (2 * c1 + c3)
    return term1 + term2 + term3
end

function compute_lambda_2(
    norm_z::Float64,
    x1::Float64,
    y1::Float64,
    z1::Float64,
    x_plus_y_dot_z::Float64,
    lambda::Float64,
    delta::Float64,
    gamma::Float64,
    c1::Float64,
    c2::Float64,
    c3::Float64,
)::Float64
    term1 = 2 * (c1 + c3) * (c2 - z1 * lambda / gamma)
    term2 = 2 * z1 * delta / gamma * (z1 * lambda / gamma - c2)
    term3 = norm_z^2 - z1^2 + 2 * (x_plus_y_dot_z - (x1 + y1) * z1)
    return term1 + term2 + term3
end

function compute_lambda_3(z1::Float64, lambda::Float64, gamma::Float64, c2::Float64)::Float64
    return c2^2 + (z1 * lambda / gamma)^2 - 2 * c2 * z1 * lambda / gamma
end

function compute_lambda_4(
    x1::Float64, y1::Float64, z1::Float64, delta::Float64, gamma::Float64, c1::Float64, c3::Float64
)::Float64
    return 2 * y1 * (c1 + c3) - 2 * z1 * delta / gamma * (x1 + y1) + 2 * c3 * x1
end

function compute_lambda_5(
    x1::Float64, y1::Float64, z1::Float64, lambda::Float64, gamma::Float64, c2::Float64
)::Float64
    return 2 * (x1 + y1) * (c2 - z1 * lambda / gamma)
end

function compute_lambda_6(x1::Float64, y1::Float64)::Float64
    return y1^2 + 2 * x1 * y1
end

function compute_all_lambdas(
    norm_x::Float64,
    norm_y::Float64,
    norm_z::Float64,
    x1::Float64,
    y1::Float64,
    z1::Float64,
    x_dot_y::Float64,
    x_plus_y_dot_z::Float64,
    lambda::Float64,
    delta::Float64,
    gamma::Float64,
    c1::Float64,
    c2::Float64,
    c3::Float64,
    horizon::Float64,
)::Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64}
    Λ0 = compute_lambda_0(c1, norm_x, horizon)
    Λ1 = compute_lambda_1(norm_y, x1, y1, z1, x_dot_y, delta, gamma, c1, c3, horizon)
    Λ2 = compute_lambda_2(norm_z, x1, y1, z1, x_plus_y_dot_z, lambda, delta, gamma, c1, c2, c3)
    Λ3 = compute_lambda_3(z1, lambda, gamma, c2)
    Λ4 = compute_lambda_4(x1, y1, z1, delta, gamma, c1, c3)
    Λ5 = compute_lambda_5(x1, y1, z1, lambda, gamma, c2)
    Λ6 = compute_lambda_6(x1, y1)

    return Λ0, Λ1, Λ2, Λ3, Λ4, Λ5, Λ6
end
