import functools
import jax
import jax.numpy as jnp
from jax import random, vmap, nn, jit
import genjax
from genjax import ChoiceMapBuilder as C
from genjax import gen, mv_normal

# Import generic EEVI estimators
from src.eevi.estimator import estimate_entropy_bounds_generic

FloatArray = jax.Array
PRNGKey = jax.random.PRNGKey
Arguments = tuple
Score = FloatArray
Weight = FloatArray


# ----------------------------------------
# Model definition: p(z) = MVN(z | mu_full, cov_full)
# This is the P-MODEL for EEVI.
# ----------------------------------------
@gen
def joint_model(mu_full: FloatArray, cov_full: FloatArray) -> FloatArray:
    """p(z) = MVN(z | mu_full, cov_full)"""
    z = mv_normal(mu_full, cov_full) @ "z"
    return z


# ----------------------------------------
# Base proposal distribution: q0(x) = p(z_{\setminus i})
# (x represents z_{\setminus i})
# This is the Q0-MODEL for EEVI.
# ----------------------------------------
@gen
def base_proposal(mu_x: FloatArray, cov_xx: FloatArray) -> FloatArray:
    """Base proposal distribution q0(x) = MVN(x | mu_x, cov_xx)"""
    x = mv_normal(mu_x, cov_xx) @ "x"
    return x


# q0 = base_proposal # q0_model will be base_proposal directly


# --- Helper functions for generic EEVI estimator for MVN ---


# reconstruct_z_values_fn: (x_k_from_q0, y_sample, target_idx) -> z_values_array
@jit
def reconstruct_z_mvn(
    x_k: FloatArray, y_sample: FloatArray, target_idx: int
) -> FloatArray:
    """Inserts scalar y_sample into x_k at target_idx to form z_k."""
    y_val = y_sample.reshape(())  # Ensure y_sample is scalar for insertion
    return jnp.insert(x_k, target_idx, y_val)


# build_p_choices_fn: (z_values_array) -> ChoiceMap for p_model (joint_model)
def build_p_choices_for_mvn(z_values_array: jnp.ndarray) -> genjax.ChoiceMap:
    return C["z"].set(z_values_array)


# extract_y_prime_fn: (joint_choice_from_p, target_idx) -> y_prime_sample_array
def extract_y_prime_for_mvn(
    joint_choice: genjax.ChoiceMap, target_idx: int
) -> jnp.ndarray:
    z_prime = joint_choice["z"]
    return z_prime[target_idx : target_idx + 1]  # Keep as array e.g. (1,)


# extract_x_prime_values_fn: (joint_choice_from_p, target_idx) -> x_prime_values_array
def extract_x_prime_values_for_mvn(
    joint_choice: genjax.ChoiceMap, target_idx: int
) -> jnp.ndarray:
    z_prime = joint_choice["z"]
    return jnp.delete(z_prime, target_idx, axis=0)


# build_q0_choices_from_x_prime_fn: (x_prime_values_array) -> ChoiceMap for q0_model (base_proposal)
def build_q0_choices_for_mvn(x_prime_values_array: jnp.ndarray) -> genjax.ChoiceMap:
    return C["x"].set(x_prime_values_array)


# sample_y_fn: (key, n_samples, mu_y, sigma_y) -> y_samples_array
def sample_y_for_mvn_entropy(
    key: PRNGKey, n_samples: int, mu_y: float, sigma_y: float
) -> jnp.ndarray:
    # Returns array of shape (n_samples,). reconstruct_z_mvn handles individual scalar y_sample.
    return random.normal(key, shape=(n_samples,)) * sigma_y + mu_y


# analytical_H_Y_fn: (variance_of_Zi) -> analytical_H_Zi
def analytical_H_Zi_mvn(var_i: float) -> float:
    return 0.5 * (1 + jnp.log(2 * jnp.pi * var_i))


# JIT compile the generic estimation function for this specific MVN case
# Helper functions are simple, JITting the main estimator should be effective.
# The helper functions passed to estimate_entropy_bounds_generic are Python callables.
# JAX's JIT will trace through them when estimate_entropy_bounds_generic_jit is first called.
# We make n_outer_samples and P_sir_particles static for JIT.
estimate_entropy_bounds_generic_jit = jit(
    estimate_entropy_bounds_generic,
    static_argnames=(
        "n_outer_samples",
        "P_sir_particles",
        "p_model",
        "q0_model",  # Models are static
        "reconstruct_z_values_fn",
        "build_p_choices_fn",
        "extract_y_prime_fn",
        "extract_x_prime_values_fn",
        "build_q0_choices_from_x_prime_fn",
        "sample_y_fn",
        "analytical_H_Y_fn",  # All helper functions are static
    ),
)


def test_mvn_entropy():
    """Runs a simplified demo for estimating a single marginal MVN entropy."""
    key = random.PRNGKey(42)  # Fixed seed for reproducibility

    # --- Demo Configuration ---
    dim_z = 2
    target_idx_to_estimate = 0  # Estimate H(Z_0)
    n_outer_samples = 10000  # Number of samples for E[...] (outer loop)
    P_sir_particles = 100  # Number of particles for SIR (inner loop)

    # Define simple MVN parameters for p(z) = joint_model(z | mu_full, cov_full)
    mu_full = jnp.array([0.5, -0.5], dtype=jnp.float32)
    cov_full = jnp.array([[1.0, 0.7], [0.7, 1.5]], dtype=jnp.float32)
    p_model_args = (mu_full, cov_full)

    # Define parameters for q0(x) = base_proposal(x | mu_x, cov_xx)
    # x = z_{\setminus target_idx_to_estimate}
    aux_indices = jnp.array([j for j in range(dim_z) if j != target_idx_to_estimate])
    mu_x_q0 = mu_full[aux_indices]
    cov_xx_q0 = cov_full[jnp.ix_(aux_indices, aux_indices)]
    q0_model_args = (mu_x_q0, cov_xx_q0)

    # Prepare curried helper functions with target_idx
    reconstruct_z_fn_curried = functools.partial(
        reconstruct_z_mvn, target_idx=target_idx_to_estimate
    )
    extract_y_prime_fn_curried = functools.partial(
        extract_y_prime_for_mvn, target_idx=target_idx_to_estimate
    )
    extract_x_prime_values_fn_curried = functools.partial(
        extract_x_prime_values_for_mvn, target_idx=target_idx_to_estimate
    )

    # Arguments for sampling y ~ p(z_i)
    mu_y_sampling = mu_full[target_idx_to_estimate]
    var_y_sampling = cov_full[target_idx_to_estimate, target_idx_to_estimate]
    sigma_y_sampling = jnp.sqrt(var_y_sampling)
    y_sampling_args_mvn = (mu_y_sampling, sigma_y_sampling)

    # Arguments for analytical H(Z_i)
    analytical_H_Y_args_mvn = (var_y_sampling,)

    print(
        "--- Simplified MVN Marginal Entropy Estimation Demo (using generic EEVI) ---"
    )
    print(f"Estimating H(Z_{target_idx_to_estimate}) for a {dim_z}-D MVN:")
    print(f"  mu_full = {mu_full}")
    print(f"  cov_full = \n{cov_full}")
    print(f"  n_outer_samples = {n_outer_samples}")
    print(f"  P_sir_particles = {P_sir_particles}\n")

    # Run the estimation using the JIT-compiled generic estimator
    analytical_H, lower_bound_H, upper_bound_H = estimate_entropy_bounds_generic_jit(
        key,
        n_outer_samples,
        P_sir_particles,
        joint_model,
        p_model_args,  # p_model and its args
        base_proposal,
        q0_model_args,  # q0_model and its args
        reconstruct_z_fn_curried,  # reconstruct_z_values_fn
        build_p_choices_for_mvn,  # build_p_choices_fn
        extract_y_prime_fn_curried,  # extract_y_prime_fn
        extract_x_prime_values_fn_curried,  # extract_x_prime_values_fn
        build_q0_choices_for_mvn,  # build_q0_choices_from_x_prime_fn
        sample_y_for_mvn_entropy,  # sample_y_fn
        y_sampling_args_mvn,  # y_sampling_args
        analytical_H_Y_fn=analytical_H_Zi_mvn,
        analytical_H_Y_args=analytical_H_Y_args_mvn,
    )

    print(
        f"  Analytical H(Z_{target_idx_to_estimate}) = {analytical_H:.4f} nats"
    )  # Note: MVN entropy is in nats
    print(
        f"  EEVI Lower Bound H_tilde(Z_{target_idx_to_estimate}) = {lower_bound_H:.4f} nats"
    )
    print(
        f"  EEVI Upper Bound H_hat(Z_{target_idx_to_estimate}) = {upper_bound_H:.4f} nats"
    )

    assert (
        lower_bound_H <= analytical_H <= upper_bound_H
    ), "The analytical entropy should be between the lower and upper bounds."
