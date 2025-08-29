"""
    sdid(data, Y_col, S_col, T_col, D_col; kwargs...)

Synthetic Difference-in-Differences (SDID) Estimator

This function implements the synthetic difference-in-differences estimator proposed by 
Arkhangelsky et al. (2021). The SDID estimator combines the synthetic control method
with difference-in-differences to estimate causal effects in panel data settings.

# Algorithm Overview:
The SDID estimator constructs synthetic control weights for both units (ω) and time 
periods (λ) to create a counterfactual for treated units in the post-treatment period.
It solves two constrained optimization problems to find weights that minimize 
pre-treatment prediction error while satisfying sum-to-one constraints.

# Parameters:
- `data`: Panel dataset containing outcome, unit, time, and treatment variables
- `Y_col`: Column name for the outcome variable
- `S_col`: Column name for unit identifiers  
- `T_col`: Column name for time periods
- `D_col`: Column name for treatment indicator (1 if treated, 0 if control)
- `covariates`: Optional vector of covariate column names
- `cov_method`: Method for handling covariates ("projected" or "optimized")
- `noise_level`: Regularization parameter (auto-calculated if not provided)
- `eta_omega`, `eta_lambda`: Base regularization parameters for unit and time weights
- `zeta_omega`, `zeta_lambda`: Scaled regularization parameters
- `omega_intercept`, `lambda_intercept`: Whether to include intercepts in weight optimization
- `min_decrease`: Minimum improvement threshold for optimization convergence
- `max_iter`: Maximum iterations for weight optimization
- `sparsify`: Function to sparsify weights (reduce to fewer non-zero elements)
- `max_iter_pre_sparsify`: Max iterations before applying sparsification
- `vce`: Variance-covariance estimation method

# Returns:
Dictionary containing:
- `att`: Average treatment effect on the treated
- `year_params`: Treatment effect estimates by treatment year
- `weights`: Estimated unit (omega) and time (lambda) weights
- `Y`: Outcome matrices by treatment year
- Additional diagnostic information
"""
function sdid(
  data, Y_col::Union{String,Symbol}, S_col::Union{String,Symbol},
  T_col::Union{String,Symbol}, D_col::Union{String,Symbol};
  covariates::Union{Vector{String},Vector{Symbol},Nothing}=nothing,
  cov_method="optimized", noise_level::Union{Float64,Nothing}=nothing,
  eta_omega::Union{Float64,Nothing}=nothing, eta_lambda::Union{Float64,Nothing}=1e-6,
  zeta_omega::Union{Float64,Nothing}=nothing, zeta_lambda::Union{Float64,Nothing}=nothing,
  omega_intercept::Bool=true, lambda_intercept::Bool=true,
  min_decrease::Union{Float64,Nothing}=nothing, max_iter::Int=10000,
  sparsify::Union{Function,Nothing}=sparsify_function,
  max_iter_pre_sparsify::Int=100, vce="placebo"
)

  # Initialize vector to store weighted treatment effects for each treatment year
  att = Float64[]

  # Convert column names to symbols for consistent indexing
  Y_col = Symbol(Y_col)
  S_col = Symbol(S_col)
  T_col = Symbol(T_col)
  D_col = Symbol(D_col)

  # Prepare the dataset: either use pre-processed data or set it up
  if all(in.(["tunit", "ty", "tyear"], Ref(names(data))))
    # Data already has required columns (tunit=treatment indicator, tyear=treatment year)
    tdf = copy(data)
    sort!(tdf, [T_col, :tunit, S_col])
  else
    # Transform data to required format with treatment indicators and years
    tdf = data_setup(data, S_col, T_col, D_col)
  end

  # Extract key dimensions and setup output containers
  t_span = collect(minimum(data[:, T_col]):maximum(data[:, T_col]))  # Full time span
  tyears = sort(unique(tdf.tyear)[.!isnothing.(unique(tdf.tyear))])  # Treatment years (excluding never-treated)
  T_total = sum(Matrix(unstack(tdf, S_col, T_col, D_col)[:, 2:end]))  # Total treated observations
  units = unique(tdf[:, S_col])  # All units
  N_out = size(units, 1)  # Total number of units
  N0_out = size(unique(tdf[tdf.tunit.==0, S_col]), 1)  # Number of control units
  T0_out = DataFrame()  # Pre-treatment periods by treatment year

  # Store original data and initialize output containers
  tdf_ori = copy(tdf)
  info_names = ["treat_year", "tau", "weighted_tau", "N0", "T0", "N1", "T1"]
  year_params = DataFrame([[] for i in info_names], info_names)  # Results by year
  year_weights = Dict("omega" => DataFrame([units[1:N0_out]], [S_col]), "lambda" => Dict())  # Weight storage
  T_out = size(unique(tdf[:, T_col]), 1)  # Number of time periods
  Y_out = Dict()  # Outcome matrices by treatment year
  info_beta = nothing  # Covariate coefficient storage

  # Handle covariates: disable covariate method if no covariates provided
  if isnothing(covariates)
    cov_method = nothing
  end
  X_out = nothing  # Covariate matrices storage

  # === PROJECTED COVARIATES METHOD ===
  # Pre-residualize outcome on covariates before applying SDID
  if !isnothing(covariates) && cov_method == "projected"
    covariates = Symbol.(covariates)

    # Project out covariates: Y_residual = Y - X*beta
    tdf, beta, X_out = projected(tdf, Y_col, S_col, T_col, covariates)
    beta_vars = [:time; covariates]
    beta = ["projected"; beta]
    info_beta = DataFrame([[] for i in beta_vars], beta_vars)
    push!(info_beta, beta)
  end

  # === MAIN ESTIMATION LOOP ===
  # Estimate treatment effects for each treatment year separately
  if isnothing(covariates) || cov_method == "projected"

    # Loop over each treatment year
    for year in tyears
      info = []
      # Extract data for current treatment year (treated + never-treated units)
      df_y = tdf[in.(tdf.tyear, Ref([year, nothing])), [Y_col, S_col, T_col, :tunit]]

      # Calculate dimensions for this treatment year
      N1 = size(unique(df_y[df_y.tunit.==1, S_col]), 1)  # Treated units
      T1 = maximum(data[:, T_col]) - year + 1  # Post-treatment periods
      T_post = N1 * T1  # Total treated observations for weighting

      # === MATRIX CONSTRUCTION ===
      # Create outcome matrix Y: rows=units, columns=time periods
      Y = Matrix(unstack(df_y, S_col, T_col, Y_col)[:, 2:end])
      N, T = size(Y)
      N0 = N - N1  # Control units
      T0 = T - T1  # Pre-treatment periods
      T0_out[:, string(year)] = [T0]

      # Collapse to block form for optimization:
      # Yc = [Y₀₀ Y₀₁; Y₁₀ Y₁₁] where 0=control/pre, 1=treated/post
      Yc = collapse_form(Y, N0, T0)

      # === REGULARIZATION PARAMETERS ===
      # Calculate noise level from pre-treatment control outcomes
      noise_level = std(diff(Y[1:N0, 1:T0], dims=2))  # Standard deviation of first differences

      # Adaptive regularization based on problem dimensions
      eta_omega = ((size(Y, 1) - N0) * (size(Y, 2) - T0))^(1 / 4)  # Scales with √(N1*T1)
      eta_lambda = 1e-6  # Small base regularization for time weights
      zeta_omega = eta_omega * noise_level    # Unit weight regularization
      zeta_lambda = eta_lambda * noise_level  # Time weight regularization
      min_decrease = 1e-5 * noise_level      # Convergence tolerance

      # === TIME WEIGHT OPTIMIZATION (λ) ===
      # Find λ that minimizes ||Y₀₀λ - Y₀₁||² + zeta_lambda * ||λ||²
      # Subject to: sum(λ) = 1, λ ≥ 0
      lambda_opt = sc_weight_fw(
        Yc[1:N0, 1:T0], Yc[1:N0, end], nothing,  # (pre-treatment outcomes, post outcomes, initial weights)
        intercept=lambda_intercept,
        zeta=zeta_lambda,
        min_decrease=min_decrease,
        max_iter=max_iter_pre_sparsify
      )

      # Apply sparsification to reduce number of non-zero weights
      if !isnothing(sparsify)
        lambda_opt = sc_weight_fw(
          Yc[1:N0, 1:T0], Yc[1:N0, end], sparsify(lambda_opt["params"]),
          intercept=lambda_intercept,
          zeta=zeta_lambda,
          min_decrease=min_decrease,
          max_iter=max_iter
        )
      end

      lambda = lambda_opt["params"]

      # === UNIT WEIGHT OPTIMIZATION (ω) ===
      # Find ω that minimizes ||ω'Y₀₀ - Y₁₀||² + zeta_omega * ||ω||²
      # Subject to: sum(ω) = 1, ω ≥ 0
      omega_opt = sc_weight_fw(
        Yc'[1:T0, 1:N0], Yc[end, 1:T0], nothing,  # (transposed for unit dimension)
        intercept=omega_intercept,
        zeta=zeta_omega,
        min_decrease=min_decrease,
        max_iter=max_iter_pre_sparsify
      )

      # Apply sparsification to unit weights
      if !isnothing(sparsify)
        omega_opt = sc_weight_fw(
          Yc'[1:T0, 1:N0], Yc[end, 1:T0], sparsify(omega_opt["params"]),
          intercept=omega_intercept,
          zeta=zeta_omega,
          min_decrease=min_decrease,
          max_iter=max_iter
        )
      end

      omega = omega_opt["params"]

      # === TREATMENT EFFECT CALCULATION ===
      # SDID estimator: τ̂ = (ω̃'Y λ̃) where ω̃ = [-ω; 1/N1 * 1_{N1}], λ̃ = [-λ; 1/T1 * 1_{T1}]
      # This computes: Y₁₁ - ω'Y₀₁ - λ'Y₁₀ + ω'Y₀₀λ
      tau_hat = [-omega; fill(1 / N1, N1)]' * Y * [-lambda; fill(1 / T1, T1)]

      # Weight by proportion of total treated observations
      tau_w = T_post / T_total * tau_hat
      att = [att; tau_w]

      # Store results for this treatment year
      info = [year tau_hat tau_w N0 T0 N1 T1]
      info_df = DataFrame([i for i in info], names(year_params))
      append!(year_params, info_df)
      year_weights["omega"][:, string(year)] = omega
      year_weights["lambda"][string(year)] = lambda
      Y_out[string(year)] = Y
    end

    year_weights["beta"] = info_beta
    # Aggregate treatment effect across all treatment years
    att = sum(att)
  end

  # === OPTIMIZED COVARIATES METHOD ===
  # Jointly optimize weights and covariate coefficients
  if cov_method == "optimized"
    beta_vars = [:time; covariates]
    info_beta = DataFrame([[] for i in beta_vars], beta_vars)
    covariates = Symbol.(covariates)
    X_out = Dict()

    # Loop over treatment years
    for year in tyears
      info = []
      # Include covariates in data extraction
      df_y = tdf[in.(tdf.tyear, Ref([year, nothing])), [[Y_col, S_col, T_col, :tunit]; covariates]]
      N1 = size(unique(df_y[df_y.tunit.==1, S_col]), 1)
      T1 = maximum(data[:, T_col]) - year + 1
      T_post = N1 * T1

      # Create outcome matrix
      Y = Matrix(unstack(df_y, S_col, T_col, Y_col)[:, 2:end])
      N = size(Y, 1)
      T = size(Y, 2)
      N0 = N - N1
      T0 = T - T1
      T0_out[:, string(year)] = [T0]
      Yc = collapse_form(Y, N0, T0)

      # Calculate regularization parameters
      noise_level = std(diff(Y[1:N0, 1:T0], dims=2))
      eta_omega = ((size(Y, 1) - N0) * (size(Y, 2) - T0))^(1 / 4)
      eta_lambda = 1e-6
      zeta_omega = eta_omega * noise_level
      zeta_lambda = eta_lambda * noise_level
      min_decrease = 1e-5 * noise_level

      # Create covariate matrices
      X = []
      for covar in covariates
        X_temp = Matrix(unstack(df_y, S_col, T_col, covar)[:, 2:end])
        push!(X, X_temp)
      end

      # Collapse covariate matrices to match outcome structure
      Xc = collapse_form.(X, N0, T0)

      # === JOINT OPTIMIZATION ===
      # Simultaneously find λ, ω, and β that minimize:
      # ||(Y - Σⱼ βⱼXⱼ)₀₀λ - (Y - Σⱼ βⱼXⱼ)₀₁||² + ||ω'(Y - Σⱼ βⱼXⱼ)₀₀ - (Y - Σⱼ βⱼXⱼ)₁₀||²
      weights = sc_weight_covariates(
        Yc, Xc, zeta_lambda=zeta_lambda, zeta_omega=zeta_omega,
        lambda_intercept=lambda_intercept, omega_intercept=omega_intercept,
        min_decrease=min_decrease, max_iter=max_iter, lambda=nothing,
        omega=nothing
      )

      # Calculate treatment effect adjusting for covariates
      X_beta = sum(weights["beta"] .* X)  # Linear combination of covariates
      tau_hat = [-weights["omega"]; fill(1 / N1, N1)]' * (Y - X_beta) * [-weights["lambda"]; fill(1 / T1, T1)]
      tau_w = T_post / T_total * tau_hat
      att = [att; tau_w]

      # Store results
      info = [year tau_hat tau_w N0 T0 N1 T1]
      info_df = DataFrame([i for i in info], names(year_params))
      append!(year_params, info_df)
      year_weights["omega"][:, string(year)] = weights["omega"]
      year_weights["lambda"][string(year)] = weights["lambda"]
      Y_out[string(year)] = Y
      X_out[string(year)] = X
      push!(info_beta, [year; weights["beta"]])
    end

    # Aggregate weighted treatment effects
    att = sum(att)
  end

  # === PREPARE OUTPUT ===
  # Return comprehensive results dictionary
  out = Dict(
    "att" => att,                    # Average treatment effect on treated
    "year_params" => year_params,    # Treatment effects by year
    "T" => T_out,                   # Number of time periods
    "N" => N_out,                   # Total units
    "N0" => N0_out,                 # Control units
    "T0" => T0_out,                 # Pre-treatment periods by year
    "data" => data,                 # Original data
    "proc_data" => tdf_ori,         # Processed data
    "tyears" => tyears,             # Treatment years
    "weights" => year_weights,      # Estimated weights (ω, λ, β)
    "Y" => Y_out,                   # Outcome matrices
    "units" => units,               # Unit identifiers
    "t_span" => t_span,             # Full time span
    "covariates" => covariates,     # Covariate names
    "cov_method" => cov_method,     # Covariate method used
    "X" => X_out,                   # Covariate matrices
    "beta" => info_beta             # Covariate coefficients
  )

  return out
end

# === COMMENTED OUT ALTERNATIVE ESTIMATORS ===
# These are alternative implementations for comparison:

# Synthetic Control (SC) estimator - uses only unit weights, no time weights
# function sc_estimate(Y, N0, T0, eta_omega=1e-6; kargs...)
#   estimate = synthdid_estimate(Y, N0, T0, eta_omega=1e-16, omega_intercept=false,
#     weights=Dict("omega" => nothing, "lambda" => fill(0, T0), "vals" => [1, 2, 3.0]))
#   return estimate
# end

# Difference-in-Differences (DID) estimator - uses equal weights for all units and times
# function did_estimate(Y, N0, T0; kargs...)
#   estimate = synthdid_estimate(Y, N0, T0, weights=Dict("omega" => fill(1 / N0, N0), "lambda" => fill(1 / T0, T0), "vals" => [1, 2, 3.0]), kargs...)
#   return estimate
# end

# === PLACEBO TESTING FUNCTIONS (TODO) ===
# These would implement placebo tests by artificially moving treatment timing

# TODO: synthdid_placebo
# function synthdid_placebo(estimate::synthdid_est1, terated_fraction=nothing)
#   setup = estimate.setup
#   opts = estimate.opts
#   weights = estimate.weight
#   x_beta = contract3(setup["X"], weights["beta"])
#   estimator = estimate.estimate

#   if (isnothing(terated_fraction))
#     terated_fraction = 1 - setup["T0"] / size(setup.Y, 2)
#   end
#   placebo_t0 = floor(setup["T0"] * (1 - terated_fraction))
# end

# Dynamic treatment effects over time
# function synthdid_effect_curve(estimate::synthdid_est1)
#   setup = estimate.setup
#   weights = estimate.weight
#   x_beta = contract3(setup["X"], weights["beta"])

#   N1 = size(setup["Y"], 1) - estimate.N0
#   T1 = size(setup["Y"], 2) - estimate.T0

#   tau_sc = vcat(-weights["omega"], fill(1 / N1, N1))' * (setup["Y"] .- x_beta)
#   tau_curve = tau_sc[setup["T0"].+(1:T1)] .- (tau_sc[1:setup["T0"]]' * weights["lambda"])
#   return tau_curve
# end

