import jax
import jax.numpy as jnp
from jax import random, vmap, nn, jit
import genjax
from genjax import ChoiceMapBuilder as C
from genjax import gen, mv_normal

FloatArray = jax.Array
PRNGKey = jax.random.PRNGKey
Arguments = tuple
Score = FloatArray
Weight = FloatArray


# ----------------------------------------
# Model definition: p(z) = MVN(z | mu_full, cov_full)
# ----------------------------------------
@gen
def joint_model(mu_full: FloatArray, cov_full: FloatArray) -> FloatArray:
    """p(z) = MVN(z | mu_full, cov_full)"""
    z = mv_normal(mu_full, cov_full) @ "z"
    return z


# ----------------------------------------
# Base proposal distribution: q0(x) = p(z_{\setminus i})
# (x represents z_{\setminus i})
# ----------------------------------------
@gen
def base_proposal(mu_x: FloatArray, cov_xx: FloatArray) -> FloatArray:
    """Base proposal distribution q0(x) = MVN(x | mu_x, cov_xx)"""
    x = mv_normal(mu_x, cov_xx) @ "x"
    return x


q0 = base_proposal


# ----------------------------------------
# Helper: Reconstruct z from x=z_{\setminus i} and y=z_i
# ----------------------------------------
@jit
def reconstruct_z(x_k: FloatArray, y_sample: FloatArray, target_idx: int) -> FloatArray:
    """Inserts scalar y_sample into x_k at target_idx to form z_k."""
    y_val = y_sample.reshape(())  # Ensure y_sample is scalar
    return jnp.insert(x_k, target_idx, y_val)


# ----------------------------------------
# SIR weight calculation for upper bound (log p(x_k, y) / q0(x_k))
# ----------------------------------------
def log_sir_weight_upper(
    key: PRNGKey,
    y_sample: FloatArray,
    model_args: Arguments,
    q0_args: Arguments,
    P_particles: int,
    target_idx: int,
) -> Weight:
    keys_q0 = random.split(key, P_particles)

    traces_q0 = vmap(q0.simulate, in_axes=(0, None))(keys_q0, q0_args)
    x_samples = vmap(lambda tr: tr.get_retval())(traces_q0)
    log_q0_vals = vmap(lambda tr: q0.assess(tr.get_choices(), q0_args)[0])(traces_q0)

    def assess_p(x_k_single_sample: FloatArray) -> Score:
        z_k = reconstruct_z(x_k_single_sample, y_sample, target_idx)
        choices = C["z"].set(z_k)
        log_p, _ = joint_model.assess(choices, model_args)
        return log_p

    log_p_vals = vmap(assess_p)(x_samples)
    log_weights = log_p_vals - log_q0_vals
    return nn.logsumexp(log_weights) - jnp.log(P_particles)


# ----------------------------------------
# SIR weight calculation for lower bound (log p(x'_k, y') / q0(x'_k))
# ----------------------------------------
def log_sir_weight_lower(
    key: PRNGKey,
    joint_choice: genjax.ChoiceMap,
    model_args: Arguments,
    q0_args: Arguments,
    P_particles: int,
    target_idx: int,
) -> Weight:
    z_prime = joint_choice["z"]
    y_prime = z_prime[target_idx : target_idx + 1]
    x_prime = jnp.delete(z_prime, target_idx, axis=0)

    log_q0_prime_1, _ = q0.assess(C["x"].set(x_prime), q0_args)
    log_p_1, _ = joint_model.assess(joint_choice, model_args)
    log_weight_1 = log_p_1 - log_q0_prime_1

    if P_particles > 1:
        keys_q0_prime_k = random.split(key, P_particles - 1)
        traces_q0_prime_k = vmap(q0.simulate, in_axes=(0, None))(
            keys_q0_prime_k, q0_args
        )
        x_prime_samples_k = vmap(lambda tr: tr.get_retval())(traces_q0_prime_k)
        log_q0_prime_vals_k = vmap(lambda tr: q0.assess(tr.get_choices(), q0_args)[0])(
            traces_q0_prime_k
        )

        def assess_p_prime_k(x_sample_k: FloatArray) -> Score:
            z_k = reconstruct_z(x_sample_k, y_prime, target_idx)
            choices = C["z"].set(z_k)
            log_p, _ = joint_model.assess(choices, model_args)
            return log_p

        log_p_vals_k = vmap(assess_p_prime_k)(x_prime_samples_k)
        log_weights_k = log_p_vals_k - log_q0_prime_vals_k
        all_log_weights = jnp.concatenate([jnp.array([log_weight_1]), log_weights_k])
    else:
        all_log_weights = jnp.array([log_weight_1])

    return nn.logsumexp(all_log_weights) - jnp.log(P_particles)


# ----------------------------------------
# Simplified EEVI entropy bound estimator for H(Z_i)
# ----------------------------------------
def estimate_single_marginal_entropy_bounds(
    key: PRNGKey,
    n_outer_samples: int,
    P_sir_particles: int,
    mu_full: FloatArray,
    cov_full: FloatArray,
    target_idx_to_estimate: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Estimates analytical, lower, and upper bounds of a single marginal entropy H(Z_i).
    """
    key_y_sampling, key_joint_sampling, key_upper_sir, key_lower_sir = random.split(
        key, 4
    )

    dim_z = mu_full.shape[0]
    model_args = (mu_full, cov_full)

    aux_indices = jnp.array([j for j in range(dim_z) if j != target_idx_to_estimate])
    mu_x_q0 = mu_full[aux_indices]
    cov_xx_q0 = cov_full[jnp.ix_(aux_indices, aux_indices)]
    q0_args = (mu_x_q0, cov_xx_q0)

    var_i = cov_full[target_idx_to_estimate, target_idx_to_estimate]
    analytical_entropy_i = 0.5 * (1 + jnp.log(2 * jnp.pi * var_i))

    # Upper bound H_hat(Z_i)
    mu_y = mu_full[target_idx_to_estimate]
    sigma_y = jnp.sqrt(var_i)
    y_keys = random.split(key_y_sampling, n_outer_samples)
    y_samples_for_upper = vmap(lambda k_y: random.normal(k_y) * sigma_y + mu_y)(
        y_keys
    ).reshape(-1, 1)

    upper_sir_keys = random.split(key_upper_sir, n_outer_samples)
    log_w_sir_upper_all = vmap(
        log_sir_weight_upper, in_axes=(0, 0, None, None, None, None)
    )(
        upper_sir_keys,
        y_samples_for_upper,
        model_args,
        q0_args,
        P_sir_particles,
        target_idx_to_estimate,
    )
    hat_H_Zi = -jnp.mean(log_w_sir_upper_all)

    # Lower bound H_tilde(Z_i)
    joint_sampling_keys = random.split(key_joint_sampling, n_outer_samples)
    joint_traces = vmap(joint_model.simulate, in_axes=(0, None))(
        joint_sampling_keys, model_args
    )
    joint_choices_for_lower = vmap(lambda tr: tr.get_choices())(joint_traces)

    lower_sir_keys = random.split(key_lower_sir, n_outer_samples)
    log_w_sir_lower_all = vmap(
        log_sir_weight_lower, in_axes=(0, 0, None, None, None, None)
    )(
        lower_sir_keys,
        joint_choices_for_lower,
        model_args,
        q0_args,
        P_sir_particles,
        target_idx_to_estimate,
    )
    tilde_H_Zi = -jnp.mean(log_w_sir_lower_all)

    return analytical_entropy_i, tilde_H_Zi, hat_H_Zi


# JIT compile the main estimation function
estimate_single_marginal_entropy_bounds_jit = jit(
    estimate_single_marginal_entropy_bounds,
    static_argnames=("n_outer_samples", "P_sir_particles", "target_idx_to_estimate"),
)


def test_mvn_entropy():
    """Runs a simplified demo for estimating a single marginal MVN entropy."""
    key = random.PRNGKey(42)  # Fixed seed for reproducibility

    # --- Demo Configuration ---
    dim_z = 2
    target_idx_to_estimate = 0  # Estimate H(Z_0)
    n_outer_samples = 1000  # Number of samples for E[...] (outer loop)
    P_sir_particles = 100  # Number of particles for SIR (inner loop)

    # Define simple MVN parameters
    mu_full = jnp.array([0.5, -0.5], dtype=jnp.float32)
    cov_full = jnp.array(
        [[1.0, 0.7], [0.7, 1.5]], dtype=jnp.float32  # Z0 variance = 1.0
    )  # Z1 variance = 1.5

    print("--- Simplified MVN Marginal Entropy Estimation Demo ---")
    print(f"Estimating H(Z_{target_idx_to_estimate}) for a {dim_z}-D MVN:")
    print(f"  mu_full = {mu_full}")
    print(f"  cov_full = \n{cov_full}")
    print(f"  n_outer_samples = {n_outer_samples}")
    print(f"  P_sir_particles = {P_sir_particles}\n")

    # Run the estimation
    analytical_H, lower_bound_H, upper_bound_H = (
        estimate_single_marginal_entropy_bounds_jit(
            key,
            n_outer_samples,
            P_sir_particles,
            mu_full,
            cov_full,
            target_idx_to_estimate,
        )
    )

    print(f"Results for H(Z_{target_idx_to_estimate}):")
    print(f"  Analytical Entropy          : {analytical_H:.4f}")
    print(f"  Estimated Lower Bound (SIR) : {lower_bound_H:.4f}")
    print(f"  Estimated Upper Bound (SIR) : {upper_bound_H:.4f}")
    print(f"  Width of Bounds             : {upper_bound_H - lower_bound_H:.4f}")
