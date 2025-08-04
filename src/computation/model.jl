using LinearAlgebra

struct DefiMarkovitzModel
    # Raw Parameters
    horizon::Float64
    mus::Vector{Float64}
    apy::Vector{Float64}
    sigmas::Vector{Float64}
    correlations::Matrix{Float64}
    liqthreshs::Vector{Float64}
    liqbonus::Vector{Float64}
    weights::Vector{Float64}
    
    # Intermediate computed Parameters
    longs::Vector{Float64}
    shorts::Vector{Float64}
    sqrt_correlations::Matrix{Float64}
end

function DefiMarkovitzModel(
    horizon::Float64,
    mus::Vector{Float64},
    apy::Vector{Float64},
    sigmas::Vector{Float64},
    correlations::Matrix{Float64},
    liqthreshs::Vector{Float64},
    liqbonus::Vector{Float64},
    weights::Vector{Float64}
)::DefiMarkovitzModel
    check =  minimum(eigvals(correlations)) > 0 ? true : throw(
        DomainError(correlations, "Correlations matrix is not positive definite")
    )
    return DefiMarkovitzModel(
        horizon,
        mus,
        apy,
        sigmas,
        correlations,
        liqthreshs,
        liqbonus,
        weights,
        max.(weights, 0),
        max.(-weights, 0),
        sqrt(correlations)
    )
end

function compute_nu_delta_lambda_gamma(model::DefiMarkovitzModel; verbose = true)::Tuple{Vector{Float64}, Float64, Float64, Float64}
    nu = model.sigmas .* (model.longs .* model.liqthreshs .- model.shorts)
    delta = sum(model.longs .* model.liqthreshs .- model.shorts)
    lambda = sum((model.longs .* model.liqthreshs .- model.shorts) .* model.mus)
    gamma = norm(model.sqrt_correlations * nu, 2)
    if verbose
        @info "Successfully computed nu, delta, lambda, gamma\n   --> nu = $nu\n   --> delta = $delta\n   --> lambda = $lambda\n   --> gamma = $gamma"
    end
    return nu, delta, lambda, gamma
end

function compute_psi_phi(model::DefiMarkovitzModel; verbose = true)::Tuple{Float64, Float64}
    avg_liq_bonus = sum(model.longs .* model.liqbonus) / sum(model.longs)
    sum_longs = sum(model.longs)
    sum_shorts = sum(model.shorts)
    ψ = 1 - sum_shorts / sum_longs * avg_liq_bonus
    ϕ = sum_shorts * (1 - avg_liq_bonus)
    if verbose
        @info "Successfully computed psi and phi\n   --> psi = $ψ\n   --> phi = $ϕ"
    end
    return ψ, ϕ
end
