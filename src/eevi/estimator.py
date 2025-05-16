import jax
import jax.numpy as jnp
from jax import random, vmap, nn
import genjax  # For type hints

# Type Aliases
FloatArray = jax.Array
PRNGKey = jax.random.PRNGKey
Arguments = tuple  # For model arguments
Score = FloatArray
Weight = FloatArray
GenFn = genjax.GenerativeFunction  # Type hint for GenJAX generative functions
ChoiceMap = genjax.ChoiceMap  # Type hint for GenJAX choice maps


def log_sir_weight_upper_generic(
    key: PRNGKey,
    y_sample,  # Sample(s) from the marginal P(Y)
    p_model: GenFn,
    p_model_args: Arguments,
    q0_model: GenFn,
    q0_model_args: Arguments,
    reconstruct_z_values_fn,  # (x_sample_from_q0, y_sample) -> z_values (tuple or array)
    build_p_choices_fn,  # (z_values) -> ChoiceMap for p_model
    P_particles: int,
) -> Weight:
    """
    Calculates the log Self-Importance Sampling (SIR) weight for the upper bound
    of entropy H_hat(Y) = -E_{y~P(Y)}[log E_{x~q0(x)}[p(x,y)/q0(x)]].
    This function computes the inner expectation for a single y_sample.
    log E_{x~q0(x)}[p(reconstruct_z(x,y))/q0(x)]
    """
    keys_q0 = random.split(key, P_particles)

    # Simulate x_k ~ q0(x) and get their log probabilities log q0(x_k)
    traces_q0 = vmap(q0_model.simulate, in_axes=(0, None))(keys_q0, q0_model_args)
    x_samples_from_q0 = vmap(lambda tr: tr.get_retval())(traces_q0)
    log_q0_vals = vmap(lambda tr: q0_model.assess(tr.get_choices(), q0_model_args)[0])(
        traces_q0
    )

    # For each x_k, reconstruct z_k from (x_k, y_sample) and assess log p(z_k)
    def assess_p_for_z_k(x_k_single_sample) -> Score:
        z_values_for_p = reconstruct_z_values_fn(x_k_single_sample, y_sample)
        p_choices = build_p_choices_fn(z_values_for_p)
        log_p, _ = p_model.assess(p_choices, p_model_args)
        return log_p

    log_p_vals = vmap(assess_p_for_z_k)(x_samples_from_q0)

    # Calculate log weights: log(p(z_k)/q0(x_k)) = log p(z_k) - log q0(x_k)
    log_weights = log_p_vals - log_q0_vals

    # Log-sum-exp for stable E[w_k] calculation: log( (1/P) * sum(weights_k) )
    return nn.logsumexp(log_weights) - jnp.log(P_particles)


def log_sir_weight_lower_generic(
    key: PRNGKey,
    joint_choice_from_p: ChoiceMap,  # A single ChoiceMap from p_model.simulate(key, p_model_args)
    p_model: GenFn,
    p_model_args: Arguments,
    q0_model: GenFn,
    q0_model_args: Arguments,
    reconstruct_z_values_fn,  # (x_sample_from_q0, y_prime_sample) -> z_values
    build_p_choices_fn,  # (z_values) -> ChoiceMap for p_model
    extract_y_prime_fn,  # (joint_choice_from_p) -> y_prime_sample
    extract_x_prime_values_fn,  # (joint_choice_from_p) -> x_prime_values
    build_q0_choices_from_x_prime_fn,  # (x_prime_values) -> ChoiceMap for q0_model
    P_particles: int,
) -> Weight:
    """
    Calculates the log Self-Importance Sampling (SIR) weight for the lower bound
    of entropy H_tilde(Y) = -E_{z'~P(Z)}[log E_{x~q0(x)}[p(x,y')/q0(x)]].
    This function computes the inner expectation for a single z' (joint_choice_from_p),
    where y' is extracted from z'.
    log E_{x~q0(x U {x'})}[p(reconstruct_z(x,y'))/q0(x)]
    The first particle is x', others are from q0.
    """
    # Extract y' (marginal sample) and x'_values (proposal components) from z' ~ p(z)
    y_prime = extract_y_prime_fn(joint_choice_from_p)
    x_prime_values = extract_x_prime_values_fn(joint_choice_from_p)

    # Build ChoiceMap for q0_model from x_prime_values to assess q0(x')
    x_prime_q0_choices = build_q0_choices_from_x_prime_fn(x_prime_values)

    # Calculate log q0(x')
    log_q0_prime_1, _ = q0_model.assess(x_prime_q0_choices, q0_model_args)
    # Calculate log p(z') (z' is the original joint sample from p_model)
    log_p_1, _ = p_model.assess(joint_choice_from_p, p_model_args)

    # First log weight term: log(p(z')/q0(x'))
    log_weight_1 = log_p_1 - log_q0_prime_1

    if P_particles > 1:
        keys_q0_prime_k = random.split(key, P_particles - 1)
        # Simulate P-1 samples x_k ~ q0(x)
        traces_q0_k = vmap(q0_model.simulate, in_axes=(0, None))(
            keys_q0_prime_k, q0_model_args
        )
        x_samples_k_from_q0 = vmap(lambda tr: tr.get_retval())(traces_q0_k)
        log_q0_vals_k = vmap(
            lambda tr: q0_model.assess(tr.get_choices(), q0_model_args)[0]
        )(traces_q0_k)

        # For each x_k, reconstruct z_k = (x_k, y') and assess p(z_k)
        def assess_p_for_zk_lower(x_k_single_sample) -> Score:
            z_values_for_p = reconstruct_z_values_fn(x_k_single_sample, y_prime)
            p_choices = build_p_choices_fn(z_values_for_p)
            log_p_val, _ = p_model.assess(p_choices, p_model_args)
            return log_p_val

        log_p_vals_k = vmap(assess_p_for_zk_lower)(x_samples_k_from_q0)
        # Log weights for these P-1 samples: log(p(z_k)/q0(x_k))
        log_weights_k = log_p_vals_k - log_q0_vals_k

        all_log_weights = jnp.concatenate([jnp.array([log_weight_1]), log_weights_k])
    else:  # P_particles == 1
        all_log_weights = jnp.array([log_weight_1])

    return nn.logsumexp(all_log_weights) - jnp.log(P_particles)


def estimate_entropy_bounds_generic(
    key: PRNGKey,
    n_outer_samples: int,
    P_sir_particles: int,
    p_model: GenFn,
    p_model_args: Arguments,
    q0_model: GenFn,
    q0_model_args: Arguments,
    reconstruct_z_values_fn,
    build_p_choices_fn,
    extract_y_prime_fn,
    extract_x_prime_values_fn,
    build_q0_choices_from_x_prime_fn,
    sample_y_fn,  # (key, n_samples, *y_sampling_args) -> y_samples (array or tuple of arrays)
    y_sampling_args: Arguments,
    # Optional: For analytical H(Y) calculation
    analytical_H_Y_fn=None,  # (*analytical_H_Y_args) -> analytical_H_Y_value
    analytical_H_Y_args: Arguments = (),
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Estimates entropy bounds H_tilde(Y) <= H(Y) <= H_hat(Y) using EEVI.

    Returns:
        A tuple (analytical_H_Y, tilde_H_Y, hat_H_Y).
        analytical_H_Y is -1.0 if not computed.
    """
    key_y_sampling, key_joint_sampling, key_upper_sir, key_lower_sir = random.split(
        key, 4
    )

    # --- Upper bound H_hat(Y) ---
    # y_samples_for_upper can be a single JAX array or a tuple of JAX arrays.
    y_samples_for_upper = sample_y_fn(key_y_sampling, n_outer_samples, *y_sampling_args)
    upper_sir_keys = random.split(key_upper_sir, n_outer_samples)

    # Determine in_axes for y_sample based on its structure
    if isinstance(y_samples_for_upper, tuple):
        # y_sample is a tuple of arrays, e.g., ((s1,s2,...), (wk1,wk2,...))
        y_sample_in_axes = tuple(0 for _ in y_samples_for_upper)
    else:
        # y_sample is a single array
        y_sample_in_axes = 0

    vmapped_log_sir_upper = vmap(
        log_sir_weight_upper_generic,
        in_axes=(
            0,  # key
            y_sample_in_axes,  # y_sample
            None,  # p_model
            None,  # p_model_args
            None,  # q0_model
            None,  # q0_model_args
            None,  # reconstruct_z_values_fn
            None,  # build_p_choices_fn
            None,  # P_particles
        ),
    )
    log_w_sir_upper_all = vmapped_log_sir_upper(
        upper_sir_keys,
        y_samples_for_upper,
        p_model,
        p_model_args,
        q0_model,
        q0_model_args,
        reconstruct_z_values_fn,
        build_p_choices_fn,
        P_sir_particles,
    )
    hat_H_Y = -jnp.mean(log_w_sir_upper_all)

    # --- Lower bound H_tilde(Y) ---
    joint_sampling_keys = random.split(key_joint_sampling, n_outer_samples)
    # Simulate z' ~ p(z)
    joint_traces_for_lower = vmap(p_model.simulate, in_axes=(0, None))(
        joint_sampling_keys, p_model_args
    )
    joint_choices_for_lower = vmap(lambda tr: tr.get_choices())(joint_traces_for_lower)
    lower_sir_keys = random.split(key_lower_sir, n_outer_samples)

    vmapped_log_sir_lower = vmap(
        log_sir_weight_lower_generic,
        in_axes=(
            0,  # key
            0,  # joint_choice_from_p
            None,  # p_model
            None,  # p_model_args
            None,  # q0_model
            None,  # q0_model_args
            None,  # reconstruct_z_values_fn
            None,  # build_p_choices_fn
            None,  # extract_y_prime_fn
            None,  # extract_x_prime_values_fn
            None,  # build_q0_choices_from_x_prime_fn
            None,  # P_particles
        ),
    )
    log_w_sir_lower_all = vmapped_log_sir_lower(
        lower_sir_keys,
        joint_choices_for_lower,
        p_model,
        p_model_args,
        q0_model,
        q0_model_args,
        reconstruct_z_values_fn,
        build_p_choices_fn,
        extract_y_prime_fn,
        extract_x_prime_values_fn,
        build_q0_choices_from_x_prime_fn,
        P_sir_particles,
    )
    tilde_H_Y = -jnp.mean(log_w_sir_lower_all)

    analytical_H_Y_val = jnp.array(-1.0, dtype=jnp.float32)  # Ensure float type
    if analytical_H_Y_fn is not None:
        analytical_H_Y_val = analytical_H_Y_fn(*analytical_H_Y_args)

    return analytical_H_Y_val, tilde_H_Y, hat_H_Y
