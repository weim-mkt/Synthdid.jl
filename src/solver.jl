"""
This file contains the core optimization routines for the Synthdid.jl package.
These functions solve for the synthetic control weights (omega and lambda) that are
central to the synthetic difference-in-differences estimator.

The primary algorithm used is the Frank-Wolfe algorithm (also known as the conditional
gradient method), which is well-suited for constrained optimization problems over a
simplex (i.e., finding weights that are non-negative and sum to one).
"""

# When fw_step and sc_weight_fw are used with omega, the columns must represent units and rows must represent time.
# The opposite is true for lambda

"""
    fw_step(A, b, x; eta, alpha=nothing)

Performs a single step of the Frank-Wolfe algorithm.

This function solves for the next iterate in the Frank-Wolfe algorithm for minimizing
the quadratic objective function: `f(x) = ||Ax - b||² + eta * ||x||²` subject to `x` being
in the probability simplex (sum(x) = 1, x >= 0).

# Arguments
- `A::Matrix`: The matrix of predictors (e.g., pre-treatment outcomes for control units).
- `b::Vector`: The vector of outcomes to be matched (e.g., pre-treatment outcomes for the treated unit).
- `x::Vector`: The current vector of weights.
- `eta::Number`: The regularization parameter.
- `alpha::Union{Nothing,Float64}`: If provided, uses a fixed step size `alpha`. Otherwise, calculates the optimal step size.

# Returns
- `Vector{Float64}`: The updated weight vector `x` after one Frank-Wolfe step.
"""
function fw_step(
  A::Matrix, b::Vector, x::Vector; eta::Number,
  alpha::Union{Nothing,Float64}=nothing
)::Vector{Float64}
  Ax = A * x
  # Gradient of the objective function w.r.t. x: 2 * A'(Ax - b) + 2 * eta * x
  # We use half the gradient for simplicity.
  half_grad = (Ax .- b)' * A + eta * x'

  # Find the vertex of the simplex that minimizes the dot product with the gradient.
  # This is the direction of steepest descent within the feasible region (the simplex).
  i = findmin(half_grad)[2][2]

  if !isnothing(alpha)
    # Fixed step size update: move towards the optimal vertex `i`.
    x *= (1 - alpha)
    x[i] += alpha
    return x
  else
    # Optimal step size calculation
    # The direction of movement is from the current point `x` to the vertex `e_i`.
    d_x = -x
    d_x[i] = 1 - x[i]

    # If the gradient is minimized at the current position, no move is needed.
    if all(d_x .== 0)
      return x
    end

    # Calculate the optimal step size (gamma) that minimizes f(x + gamma * d_x).
    # For a quadratic, the optimal step is -gradient' * direction / (direction' * Hessian * direction).
    d_err = A[:, i] - Ax
    step_upper = -half_grad * d_x
    step_bot = sum(d_err .^ 2) + eta * sum(d_x .^ 2)
    step = step_upper[1] / step_bot

    # Constrain step to [0, 1] to stay within the simplex.
    constrained_step = min(1, max(0, step))
    return x + constrained_step * d_x
  end
end

"""
    sc_weight_fw(A, b, x; intercept, zeta, min_decrease, max_iter)

Computes synthetic control weights using the Frank-Wolfe algorithm.

This function iteratively calls `fw_step` to find the optimal weights `x` that
minimize the regularized loss function: `||Ax - b||²/n + zeta² * ||x||²`.

# Arguments
- `A::Matrix`: Predictor matrix.
- `b::Vector`: Outcome vector to match.
- `x::Union{Vector, Nothing}`: Initial weights. If `nothing`, uniform weights are used.
- `intercept::Bool`: If true, de-means the data to include an intercept.
- `zeta::Number`: Regularization parameter.
- `min_decrease::Number`: Convergence threshold for the objective function value.
- `max_iter::Int64`: Maximum number of iterations.

# Returns
- `Dict`: A dictionary containing the optimal `"params"` (weights) and the objective function `"vals"` at each iteration.
"""
function sc_weight_fw(
  A::Matrix, b::Vector, x::Union{Vector,Nothing}=nothing;
  intercept::Bool=true, zeta::Number,
  min_decrease::Number=1e-3, max_iter::Int64=1000
)

  k = size(A, 2)
  n = size(A, 1)
  if isnothing(x)
    # Initialize with uniform weights if none are provided
    x = fill(1 / k, k)
  end
  if intercept
    # De-mean data to effectively fit an intercept
    A = A .- mean(A, dims=1)
    b = b .- mean(b, dims=1)
  end

  t = 0
  vals = zeros(max_iter)
  # The `eta` in fw_step corresponds to `n * zeta^2` for the loss function `||Ax-b||^2 + eta*||x||^2`
  eta = n * real(zeta^2)

  # Iterate until convergence or max_iter is reached
  while (t < max_iter) && (t < 2 || vals[t-1] - vals[t] > min_decrease^2)
    t += 1
    x_p = fw_step(A, b, x, eta=eta)
    x = x_p
    err = A * x - b
    # Objective function value
    vals[t] = real(zeta^2) * sum(x .^ 2) + sum(err .^ 2) / n
  end
  Dict("params" => x, "vals" => vals)
end

# sc_weight_covariates only runs when there are covariates specified
# This implements the procedure as in Abadie et al. (2010)
# X::Vector{Matrix{Float64}} w/ covariates
# Y (outcome) and X must come from unstacked data (year columns, unit rows, covariate values) and must be in collapsed form
# For staggered treatments, this function should be applied for a in A (see Clarke et al. 2023, p. 9-10)

"""
    sc_weight_covariates(Y, X; kwargs...)

Computes synthetic control weights and covariate coefficients jointly.

This function implements an alternating optimization algorithm to find the unit weights
(omega), time weights (lambda), and covariate coefficients (beta) simultaneously.
It alternates between updating the weights (lambda, omega) for a fixed beta, and
taking a gradient descent step for beta with fixed weights.

This approach is used when `cov_method = "optimized"`.

# Arguments
- `Y::Matrix`: Collapsed outcome matrix `[Y₀₀ Y₀₁; Y₁₀ Y₁₁]`.
- `X::Vector`: A vector of collapsed covariate matrices.
- `zeta_lambda`, `zeta_omega`: Regularization parameters for time and unit weights.
- `lambda_intercept`, `omega_intercept`: Whether to include intercepts.
- `min_decrease`: Convergence threshold.
- `max_iter`: Maximum number of iterations.
- `lambda`, `omega`, `beta`: Optional initial values for the parameters.
- `update_lambda`, `update_omega`: Flags to control which weights are updated.

# Returns
- `Dict`: A dictionary containing `"lambda"`, `"omega"`, `"beta"`, and `"vals"`.
"""
function sc_weight_covariates(
  Y::Matrix, X::Vector; zeta_lambda=0, zeta_omega=0,
  lambda_intercept::Bool=true, omega_intercept::Bool=true,
  min_decrease::Float64=1e-3, max_iter::Int=1000,
  lambda=nothing, omega=nothing, beta=nothing,
  update_lambda::Bool=true, update_omega::Bool=true
)

  T0 = size(Y, 2) - 1
  N0 = size(Y, 1) - 1

  # Initialize parameters if not provided
  if isnothing(lambda)
    lambda = fill(1 / T0, T0)
  end
  if isnothing(omega)
    omega = fill(1 / N0, N0)
  end
  if isnothing(beta)
    beta = zeros(size(X, 1))
  end

  # Nested function to update lambda and omega for a given Y
  function update_weights(Y, lambda, omega)

    # Update time weights (lambda)
    Y_lambda = if lambda_intercept
      Y[1:N0, :] .- mean(Y[1:N0, :], dims=1)
    else
      Y[1:N0, :]
    end
    if update_lambda
      lambda = fw_step(Y_lambda[:, 1:T0], Y_lambda[:, T0+1], lambda, eta=N0 * real(zeta_lambda^2))
    end
    err_lambda = Y_lambda * [lambda; -1]

    # Update unit weights (omega)
    Y_omega = if omega_intercept
      Y'[1:T0, :] .- mean(Y'[1:T0, :], dims=1)
    else
      Y[:, 1:T0]'
    end
    if update_omega
      omega = fw_step(Y_omega[:, 1:N0], Y_omega[:, N0+1], omega, eta=T0 * real(zeta_omega^2))
    end
    err_omega = Y_omega * [omega; -1]

    # Combined objective function value
    val = real(zeta_omega^2) * sum(omega .^ 2) + real(zeta_lambda^2) * sum(lambda .^ 2) + sum(err_omega .^ 2) / T0 + sum(err_lambda .^ 2) / N0

    return Dict("val" => val, "lambda" => lambda, "omega" => omega, "err_lambda" => err_lambda, "err_omega" => err_omega)
  end

  vals = zeros(max_iter)
  t = 0

  # Start with Y residualized on initial beta
  Y_beta = Y - sum(beta .* X)
  weights = update_weights(Y_beta, lambda, omega)

  # Main optimization loop
  while (t < max_iter) && ((t < 2) || (vals[t-1] - vals[t] > min_decrease^2))

    t += 1
    # === Update Beta (Gradient Descent Step) ===
    # Gradient of the objective function w.r.t. beta
    gr_lambda = (Ref(weights["err_lambda"]') .* [arr[1:N0, :] for arr in X]) .* Ref([weights["lambda"]; -1]) ./ N0
    gr_omega = (Ref(weights["err_omega"]') .* [arr[:, 1:T0]' for arr in X]) .* Ref([weights["omega"]; -1]) ./ T0
    grad_beta = -(gr_lambda[1] + gr_omega[1])

    # Update beta with a decaying step size
    alpha = 1 / t
    beta = beta .- alpha * grad_beta

    # === Update Weights ===
    # Residualize Y on the new beta
    Y_beta = Y - sum(beta .* X)
    # Update lambda and omega with the new residualized Y
    weights = update_weights(Y_beta, weights["lambda"], weights["omega"])
    vals[t] = weights["val"]
  end
  return Dict("lambda" => weights["lambda"], "omega" => weights["omega"], "beta" => beta, "vals" => vals)
end


