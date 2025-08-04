struct modelParams 
    T::Real
    C::Matrix{<:Real} 
    μ::Vector{<:Real}
    σ::Vector{<:Real}
    r::Vector{<:Real}
    lt::Vector{<:Real}
    lb::Vector{<:Real}
end

function wrap_objective(params::modelParams, Φ::Real)::Function
    return x -> compute_objective(
        DefiMarkovitzModel(
            params.T, params.μ, params.r, params.σ, params.C, params.lt, params.lb, x
        ),
        Φ
    )
end

function compute_gradient(objective::Function, x::Vector{<:Real})::Vector{<:Real}
    d = length(x)
    g = zeros(d)
    ε = 1e-5
    for i in 1:d
        δ = zeros(d)
        δ[i] = ε
        g[i] = (objective(x + δ) - objective(x - δ)) / (2*ε)
    end
    return g
end

function orthogonal_projection(x::Vector{<:Real})#::Vector{<:Real}
    d = length(x)
    B = Bidiagonal(ones(d), - ones(d-1), :L)[:, 1:d-1]
    P = B * inv(transpose(B) * B) * transpose(B)
    return P * x
end

# function gradient_descent(objective::Function, x0::Vector{<:Real}, lr::Real, niter::Int)
#     x = x0
#     history = []
#     for _ in 1:niter
#         g = compute_gradient(objective, x)
#         x = x - lr * g
#         append!(history, x)
#     end
#     return x, history
# end

function adam_optimize(
    objective::Function, niter::Int, theta::Vector{<:Real}, lr::Real, betas::Vector{<:Real}, lambda::Real, epsilon::Real; project = false, verbose = false
)::Vector{<:Real}
    d = length(theta)
    m = zeros(d)
    v = zeros(d)
    x = theta
    beta1 = betas[1]
    beta2 = betas[2]
    score = Inf
    a = zeros(d); a[1] = 1
    for t in 1:niter
        g = compute_gradient(objective, x)
        if project
            g = orthogonal_projection(g)
        end
        g = g + lambda * x
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g.^2
        m_hat = m ./ (1 - beta1^t)
        v_hat = v ./ (1 - beta2^t)
        x = x - lr * m_hat ./ (sqrt.(v_hat) .+ epsilon)
        if project 
            x = a + orthogonal_projection(x - a)
        end
        if t%10 == 0 && verbose
            score = objective(x)
            @info "Iteration $t : current score = $score"
        end
    end
    return x
end
