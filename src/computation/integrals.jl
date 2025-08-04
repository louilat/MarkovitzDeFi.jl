using Distributions

function compute_I0(eta, delta, gamma, horizon)
    N = Normal()
    left = exp(-eta * delta / gamma) - exp(-eta * delta / gamma) * cdf(N, delta / (gamma * sqrt(horizon)) - eta * sqrt(horizon))
    right = exp(eta * delta / gamma) - exp(eta * delta / gamma) * cdf(N, delta / (gamma * sqrt(horizon)) + eta * sqrt(horizon))
    return left + right
end

function compute_I1(eta, delta, gamma, horizon)
    N = Normal()
    left = exp(eta * delta / gamma) * cdf(N, delta / (gamma * sqrt(horizon)) + eta * sqrt(horizon)) - exp(eta * delta / gamma)
    right = exp(-eta * delta / gamma) - exp(-eta * delta / gamma) * cdf(N, delta / (gamma * sqrt(horizon)) - eta * sqrt(horizon))
    return - delta / gamma * (left + right)
end

function compute_I2(eta, delta, gamma, horizon)
    N = Normal()
    coeff = delta / (eta * gamma) * exp(-eta * delta / gamma)
    left = (eta * delta / gamma + 1) * (cdf(N, delta / (gamma * sqrt(horizon)) - eta * sqrt(horizon)) - 1/2)
    right = (eta * delta / gamma - 1) * (exp(2 * eta * delta / gamma) * cdf(N, delta / (gamma * sqrt(horizon)) + eta * sqrt(horizon)) - exp(2 * eta * delta / gamma) + 1/2)
    cst =  - delta / gamma * sqrt(2 * horizon / pi) * exp(- eta^2 / 2 * horizon - (delta / gamma)^2 / (2 * horizon)) + (delta / gamma)^2 * exp(-eta * delta / gamma)
    return 1 / eta * compute_I1(eta, delta, gamma, horizon) - coeff * (left + right) + cst
end