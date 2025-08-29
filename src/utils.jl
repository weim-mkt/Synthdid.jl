"""
This file contains various helper and utility functions for the Synthdid.jl package.
These functions support data preprocessing, covariate adjustments, matrix transformations,
and synthetic data generation for testing purposes.
"""


"""
    sparsify_function(v::Vector)

Sparsifies a weight vector by setting small values to zero and renormalizing.

This function implements a simple heuristic for creating sparse weights. It sets any
weight less than or equal to one-quarter of the maximum weight to zero. The remaining
weights are then renormalized to sum to one.

# Arguments
- `v::Vector`: The input weight vector.

# Returns
- `Vector`: A sparsified and renormalized weight vector.
"""
function sparsify_function(v::Vector)
  v[v.<=maximum(v)/4] .= 0
  return v ./ sum(v)
end

"""
    data_setup(data, S_col, T_col, D_col)

Prepares a panel DataFrame for SDID estimation.

This function processes a standard panel dataset to create variables required by the
`sdid` function. It identifies treated and control units and determines the timing of
treatment adoption for staggered treatment designs.

# Key operations:
1.  `:tunit`: An indicator that is 1 for ever-treated units and 0 for never-treated units.
2.  `:ty`: The time period of treatment for a treated observation, `nothing` otherwise.
3.  `:tyear`: The first period of treatment for an ever-treated unit, `nothing` for control units.

# Arguments
- `data::DataFrame`: The input panel data.
- `S_col`: Column name for the unit identifier.
- `T_col`: Column name for the time period.
- `D_col`: Column name for the treatment indicator.

# Returns
- `DataFrame`: A new DataFrame with the added `:tunit`, `:ty`, and `:tyear` columns.
"""
function data_setup(
  data::DataFrame, S_col::Union{String,Symbol},
  T_col::Union{String,Symbol}, D_col::Union{String,Symbol}
)

  tdf = copy(data)
  # Create :tunit - indicator for being an ever-treated unit
  select!(groupby(tdf, S_col), :, D_col => maximum => :tunit)
  # Create :ty - the time of treatment, or nothing if not treated
  tdf.ty = @. ifelse(tdf[:, D_col] == 0, nothing, tdf[:, T_col])
  # Create :tyear - the first year of treatment for a unit
  select!(groupby(tdf, S_col), :, :ty => minnothing => :tyear)
  sort!(tdf, [T_col, :tunit, S_col])

  return tdf
end

"""
    projected(data, Y_col, S_col, T_col, covariates)

Adjusts the outcome variable by projecting out covariates.

This function implements the "projected" method for handling covariates. It performs
an OLS regression of the outcome on the specified covariates plus unit and time fixed
effects, using only the observations from control units. The estimated coefficients
for the covariates are then used to create a residualized outcome variable for the
entire dataset (`Y_adj = Y - X * beta`).

# Arguments
- `data`: The input DataFrame, processed by `data_setup`.
- `Y_col`, `S_col`, `T_col`: Column names for outcome, unit, and time.
- `covariates`: A vector of covariate column names.

# Returns
- A tuple `(data, beta, X)` where:
  - `data`: The DataFrame with the outcome column replaced by the adjusted outcome.
  - `beta`: The estimated OLS coefficients for the covariates.
  - `X`: The original covariate matrix.
"""
function projected(data, Y_col, S_col, T_col, covariates)

  k = size(covariates, 1)
  X = Matrix(data[:, covariates])
  y = data[:, Y_col]

  # Use only control units for the regression
  df_c = data[isnothing.(data.tyear), :]

  # Create fixed effects via one-hot encoding for units and time
  # Note: One category is dropped for each to avoid multicollinearity
  select!(df_c, :, [S_col => ByRow(isequal(v)) => Symbol(v) for v in unique(df_c[:, S_col])[2:end]])
  select!(df_c, :, [T_col => ByRow(isequal(v)) => Symbol(v) for v in unique(df_c[:, T_col])[2:end]])
  o_h_cov = Symbol.([covariates; unique(df_c[:, S_col])[2:end]; unique(df_c[:, T_col])[2:end]])

  # Create X_c Matrix with covariates, one-hot encoding for T_col and S_col. Create Y_c vector
  y_c = df_c[:, Y_col]
  X_c = Matrix(df_c[:, o_h_cov])

  # Manual OLS: beta = (X'X)^-1 * X'y
  XX = [X_c ones(size(X_c, 1))]' * [X_c ones(size(X_c, 1))]
  Xy = [X_c ones(size(X_c, 1))]' * y_c
  all_beta = inv(XX) * Xy
  # We only need the coefficients for the user-specified covariates
  beta = all_beta[1:k]

  # Calculate the adjusted outcome for all observations
  Y_adj = y - X * beta

  # Return the modified dataset
  data[:, Y_col] = Y_adj
  return data, beta, X
end

"""
    minnothing(x)

Calculates the minimum of a vector, ignoring `nothing` values.
Returns `nothing` if the vector contains only `nothing` or is empty.
"""
function minnothing(x)
  x = x[.!isnothing.(x)]
  if length(x) == 0
    return nothing
  end
  return minimum(x)
end

"""
    find_treat(W)

Counts the number of treated units from a treatment matrix `W`.
Assumes `W` is a binary matrix where `W[i, t] = 1` if unit `i` is treated at time `t`.
"""
function find_treat(W)
  N1 = 0
  # A unit is treated if it has a 1 in any time period.
  for row in eachrow(W)
    if 1 in row
      N1 += 1
    end
  end
  return N1
end

"""
    collapse_form(Y::Matrix, N0::Int64, T0::Int64)

Collapses a full outcome matrix into the 2x2 block average form.

This is a key transformation for the SDID estimator. It takes the full `N x T` outcome
matrix and computes the four block averages required for the optimization:
1.  `Y_T0N0`: Pre-treatment outcomes for control units (no averaging).
2.  `Y_T1N0`: Post-treatment outcomes for control units, averaged over time.
3.  `Y_T0N1`: Pre-treatment outcomes for treated units, averaged over units.
4.  `Y_T1N1`: Post-treatment outcomes for treated units, averaged over units and time.

# Arguments
- `Y::Matrix`: The full `N x T` outcome matrix.
- `N0::Int64`: The number of control units.
- `T0::Int64`: The number of pre-treatment periods.

# Returns
- A `(N0+1) x (T0+1)` matrix containing the collapsed block averages.
"""
function collapse_form(Y::Matrix, N0::Int64, T0::Int64)
  N, T = size(Y)
  Y_T0N0 = Y[1:N0, 1:T0]
  Y_T1N0 = mean(Y[1:N0, T0+1:end], dims=2)
  Y_T0N1 = mean(Y[N0+1:end, 1:T0], dims=1)
  Y_T1N1 = mean(Y[N0+1:end, T0+1:end])

  return [Y_T0N0 Y_T1N0; Y_T0N1 Y_T1N1]
end

"""
    pairwise_sum_decreasing(x::Vector, y::Vector)

Custom pairwise sum of two vectors, handling NaN values.
If a value is NaN in one vector, it's imputed with the minimum non-NaN value before summing.
If a value is NaN in both vectors, the result is NaN.
"""
# function pairwise_sum_decreasing(x::Vector{Number}, y::Vector{Number})
function pairwise_sum_decreasing(x::Vector, y::Vector)
  na_x = isnan.(x)
  na_y = isnan.(y)
  x[na_x] .= minimum(x[.!na_x])
  y[na_y] .= minimum(y[.!na_y])
  pairwise_sum = x .+ y
  pairwise_sum[na_x.&na_y] .= NaN
  return pairwise_sum
end

# The following functions are for generating synthetic data for testing.

"""
A mutable struct to hold synthetic panel data for testing.
"""
mutable struct random_walk
  Y::Matrix       # The outcome matrix (N x T)
  n0::Number      # Number of control units
  t0::Number      # Number of pre-treatment periods
  L::Matrix       # The underlying low-rank component of Y (noise-free)
end

"""
    random_low_rank()

Generates synthetic panel data with a low-rank structure.

This function creates a simulated dataset that is well-suited for synthetic control
methods. The data generating process includes:
- A low-rank matrix `U*V'` representing latent factors.
- Unit and time fixed effects (`alpha` and `beta`).
- A constant treatment effect `tau` added to treated units in the post-period.
- Poisson-distributed noise.

# Returns
- A `random_walk` struct containing the generated data.
"""
function random_low_rank()
  n0 = 100
  n1 = 10
  t0 = 120
  t1 = 20
  n = n0 + n1
  t = t0 + t1
  tau = 1
  sigma = 0.5
  rank = 2
  rho = 0.7
  var = [rho^(abs(x - y)) for x in 1:t, y in 1:t]
  # Treatment matrix: 1 for treated units post-treatment, 0 otherwise
  W = Int.(1:n .> n0) * transpose(Int.(1:t .> t0))

  # Create low-rank factor matrices U and V from Poisson distributions
  # U = rand(Poisson(sqrt.(1:n) ./ sqrt(n)), n, rank)
  pU = Poisson(sqrt(sample(1:n)) ./ sqrt(n))
  pV = Poisson(sqrt(sample(1:t)) ./ sqrt(t))
  U = rand(pU, n, rank)
  V = rand(pV, t, rank)

  # sample.(1:n)

  # Create unit and time fixed effects
  alpha = reshape(repeat(10 * (1:n) ./ n, outer=(t, 1)), n, t)
  beta = reshape(repeat(10 * (1:t) ./ t, outer=(n, 1)), n, t)
  # Combine components to form the noise-free outcome matrix L (mu)
  mu = U * V' + alpha + beta
  error = rand(pV, size(mu))
  # Add treatment effect and noise to create the final outcome matrix Y
  Y = mu .+ tau .* W .+ sigma .* error
  random_data = random_walk(Y, n0, t0, mu)
  return random_data
end