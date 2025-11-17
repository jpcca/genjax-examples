# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp

import genjax
from genjax import gen, normal, gamma, dirichlet, categorical, mix  # type: ignore
from genjax import ChoiceMap  # type: ignore
from genjax._src.core.compiler.interpreters.incremental import Diff  # type: ignore
from genjax import Const  # type: ignore


# ------------------------------
# Utilities
# ------------------------------
def hill(x, kd, n, eps=1e-6):
    x = jnp.maximum(x, eps)
    return 1.0 / (1.0 + (kd / x) ** n)


def log_normal_pdf(y, mu, sigma):
    return -0.5 * (
        jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(sigma) + ((y - mu) / sigma) ** 2
    )


# ------------------------------
# Components (top-level)
# ------------------------------
@gen
def hill_mu_component(xi, kd, n_k, alpha_k):
    mu = hill(xi, kd, n_k)
    mu = alpha_k * mu
    return mu


# ------------------------------
# Generative model (vectorized with mix combinator)
# ------------------------------
@gen
def hill_mixture_model(
    xs,
    alpha,
    shape_kd=100.0,
    rate_kd=5.0,
    shape_n=4.0,
    rate_n=2.0,
    sigma=0.02,
    shape_alpha=5.0,
    rate_alpha=5.0,
):
    K = 3

    # Global parameters
    w = dirichlet(alpha) @ "weights"  # (K,)
    Kd = gamma(shape_kd, rate_kd, sample_shape=Const((K,))) @ "clusters/Kd"  # (K,)
    n = gamma(shape_n, rate_n, sample_shape=Const((K,))) @ "clusters/n"  # (K,)
    Alpha = (
        gamma(shape_alpha, rate_alpha, sample_shape=Const((K,))) @ "clusters/alpha"
    )  # (K,)

    # Build 3-component mixture and vmap over observations
    logits = jnp.log(w + 1e-20)
    mix3 = mix(hill_mu_component, hill_mu_component, hill_mu_component)
    mix3_vm = mix3.vmap(
        in_axes=(
            None,  # logits shared across rows
            (0, None, None, None),  # args for comp 1: (xs, Kd[0], n[0], Alpha[0])
            (0, None, None, None),  # args for comp 2
            (0, None, None, None),  # args for comp 3
        )
    )

    mu_vec = (
        mix3_vm(
            logits,
            (xs, Kd[0], n[0], Alpha[0]),
            (xs, Kd[1], n[1], Alpha[1]),
            (xs, Kd[2], n[2], Alpha[2]),
        )
        @ "y_mix"
    )
    y = normal(mu_vec, sigma) @ "y"
    return y


# ------------------------------
# Jitted update for hill_mixture_model
# ------------------------------
jitted_update = jax.jit(hill_mixture_model.update)


# ------------------------------
# Proposals
# ------------------------------


"""Proposals using propose/assess to directly produce ChoiceMap for model.update."""


"""Random-walk proposals for globals (Kd, n) in log-space.

The prior-independent proposals led to poor mixing when data are abundant and
noise is small. We switch to symmetric random-walk proposals in log-space and
update parameters locally per-cluster and per-parameter to improve acceptance.
"""


def _mh_rw_one(key, trace, address: str, idx: int, step: float):
    """One MH random-walk update on log-parameter for a single index.

    - address: one of "clusters/Kd" or "clusters/n"
    - idx: which cluster index to update
    - step: stddev of the additive noise in log-space

    Uses model.update to compute the log target ratio; adds the Hastings
    correction (z' - z) arising from the exp transform when proposing in
    log-space.
    """
    argdiffs = Diff.no_change(trace.get_args())

    ch = trace.get_choices()
    cur_vec = ch[address]
    x_cur = cur_vec[idx]
    z_cur = jnp.log(jnp.maximum(x_cur, 1e-20))

    key, sub = jax.random.split(key)
    z_prop = z_cur + step * jax.random.normal(sub)
    x_prop = jnp.exp(z_prop)

    prop_vec = cur_vec.at[idx].set(x_prop)
    prop_cm = ChoiceMap.empty()
    prop_cm = prop_cm.at[address].set(prop_vec)

    key, sub = jax.random.split(key)
    new_trace, model_logw, _, _ = jitted_update(sub, trace, prop_cm, argdiffs)

    # Hastings correction for proposing in log-space: log q(x|x') - log q(x'|x) = z' - z
    delta_log_q = z_prop - z_cur
    alpha = model_logw + delta_log_q

    key, sub = jax.random.split(key)
    accept = jnp.log(jax.random.uniform(sub)) < alpha
    out_trace = jax.lax.cond(accept, lambda _: new_trace, lambda _: trace, operand=None)
    return key, out_trace


# Gibbs-style proposal for weights using propose/assess directly on model address
@gen
def prop_weights(trace, *model_args):
    ch = trace.get_choices()
    z_vec = ch["y_mix", "mixture_component"]
    K = ch["weights"].shape[0]
    alpha = model_args[1]
    counts = jnp.bincount(z_vec, length=K)
    post_alpha = alpha + counts
    _ = dirichlet(post_alpha) @ "weights"
    return None


@gen
def prop_z(trace, *model_args):
    """Propose z vector at ("y_mix","mixture_component") using collapsed categorical per row.
    Computes per-row logits ~ log w_k + log p(y_i | x_i, Kd_k, n_k).
    """
    ch = trace.get_choices()
    xs = trace.get_args()[0]
    y_vec = ch["y"]
    w = ch["weights"]  # (K,)
    Kd = ch["clusters/Kd"]  # (K,)
    n = ch["clusters/n"]  # (K,)
    Alpha = ch["clusters/alpha"]  # (K,)
    sigma = model_args[-1]

    # Compute per-row logits (N,K)
    mu_all = jnp.stack(
        [
            Alpha[0] * hill(xs, Kd[0], n[0]),
            Alpha[1] * hill(xs, Kd[1], n[1]),
            Alpha[2] * hill(xs, Kd[2], n[2]),
        ],
        axis=1,
    )  # (N, K)
    ll = -0.5 * (
        jnp.log(2.0 * jnp.pi * sigma**2) + ((y_vec[:, None] - mu_all) / sigma) ** 2
    )
    logits = jnp.log(w + 1e-20)[None, :] + ll  # (N,K)

    # Sample vector of indices directly at the model's mixture address
    _ = categorical(logits) @ ("y_mix", "mixture_component")
    return None


# ------------------------------
# Single-step kernels using propose/assess + model.update
# ------------------------------


def mh_step_globals_rw(key, trace, step_kd: float, step_n: float, step_alpha: float):
    """One sweep of local RW updates over all clusters for Kd, n, and alpha.

    Proposes in log-space with Normal(0, step^2) increments and includes the
    Hastings correction. Updates K components sequentially using a JAX fori_loop.
    """
    ch = trace.get_choices()
    K = ch["clusters/Kd"].shape[0]

    def body(i, carry):
        key_s, trace_s = carry
        key_s, trace_s = _mh_rw_one(key_s, trace_s, "clusters/Kd", i, step_kd)
        key_s, trace_s = _mh_rw_one(key_s, trace_s, "clusters/n", i, step_n)
        key_s, trace_s = _mh_rw_one(key_s, trace_s, "clusters/alpha", i, step_alpha)
        return (key_s, trace_s)

    key, trace = jax.lax.fori_loop(0, K, body, (key, trace))
    return key, trace


def mh_step_weights(key, trace, alpha):
    """Update weights | z via Dirichlet conditional using propose/assess.
    This is a Gibbs-style move; acceptance should be ~1 in theory, but we still
    run the generic MH accept/reject for consistency.
    """
    argdiffs = Diff.no_change(trace.get_args())
    model_args = trace.get_args()

    key, sub = jax.random.split(key)
    fwd_choices, fwd_weight, _ = prop_weights.propose(sub, (trace, *model_args))

    key, sub = jax.random.split(key)
    new_trace, model_logw, _, discard = jitted_update(sub, trace, fwd_choices, argdiffs)

    bwd_weight, _ = prop_weights.assess(discard, (new_trace, *model_args))

    alpha_log = model_logw - fwd_weight + bwd_weight
    key, sub = jax.random.split(key)
    accept = jnp.log(jax.random.uniform(sub)) < alpha_log
    out_trace = jax.lax.cond(accept, lambda _: new_trace, lambda _: trace, operand=None)
    return key, out_trace


def mh_step_z(key, trace, sigma):
    """Update z | rest using collapsed categorical via propose/assess."""
    argdiffs = Diff.no_change(trace.get_args())
    model_args = trace.get_args()

    key, sub = jax.random.split(key)
    fwd_choices, fwd_weight, _ = prop_z.propose(sub, (trace, *model_args))

    key, sub = jax.random.split(key)
    new_trace, model_logw, _, discard = jitted_update(sub, trace, fwd_choices, argdiffs)

    bwd_weight, _ = prop_z.assess(discard, (new_trace, *model_args))

    alpha = model_logw - fwd_weight + bwd_weight
    key, sub = jax.random.split(key)
    accept = jnp.log(jax.random.uniform(sub)) < alpha
    out_trace = jax.lax.cond(accept, lambda _: new_trace, lambda _: trace, operand=None)
    return key, out_trace


# ------------------------------
# Full inference loop
# ------------------------------


def run_inference(
    key,
    model,
    model_args,
    observations,
    n_rounds=200,
    mh_step_kd=0.03,
    mh_step_n=0.03,
    mh_step_alpha=0.03,
    burn_in: int = 0,
    thin: int = 1,
):
    key, sub = jax.random.split(key)
    trace, _ = model.importance(sub, observations, model_args)
    alpha = model_args[1]
    sigma = model_args[-1]

    # Prepare accumulators for posterior means (weights, Kd, n)
    ch0 = trace.get_choices()
    sum_w = jnp.zeros_like(ch0["weights"])  # (K,)
    sum_kd = jnp.zeros_like(ch0["clusters/Kd"])  # (K,)
    sum_n = jnp.zeros_like(ch0["clusters/n"])  # (K,)
    sum_alpha = jnp.zeros_like(ch0["clusters/alpha"])  # (K,)
    count = jnp.array(0, dtype=jnp.int32)

    def _step(carry, i):
        key_s, trace_s, sum_w_s, sum_kd_s, sum_n_s, sum_alpha_s, cnt_s = carry

        key_s, trace_s = mh_step_globals_rw(
            key_s, trace_s, mh_step_kd, mh_step_n, mh_step_alpha
        )
        # Always update z and weights
        key_s, trace_s = mh_step_z(key_s, trace_s, sigma)
        key_s, trace_s = mh_step_weights(key_s, trace_s, alpha)

        # Accumulate running sums with burn-in and thinning
        ch_s = trace_s.get_choices()
        take = jnp.logical_and(i >= burn_in, ((i - burn_in) % thin) == 0)
        sum_w_s = jnp.where(take, sum_w_s + ch_s["weights"], sum_w_s)
        sum_kd_s = jnp.where(take, sum_kd_s + ch_s["clusters/Kd"], sum_kd_s)
        sum_n_s = jnp.where(take, sum_n_s + ch_s["clusters/n"], sum_n_s)
        sum_alpha_s = jnp.where(take, sum_alpha_s + ch_s["clusters/alpha"], sum_alpha_s)
        cnt_s = cnt_s + jnp.where(take, 1, 0)

        return (key_s, trace_s, sum_w_s, sum_kd_s, sum_n_s, sum_alpha_s, cnt_s), None

    (key, trace, sum_w, sum_kd, sum_n, sum_alpha, count), _ = jax.lax.scan(
        _step,
        (key, trace, sum_w, sum_kd, sum_n, sum_alpha, count),
        xs=jnp.arange(n_rounds),
        length=n_rounds,
    )

    # Compute posterior means safely
    denom = jnp.maximum(count.astype(sum_w.dtype), 1)
    post_means = {
        "weights": sum_w / denom,
        "clusters/Kd": sum_kd / denom,
        "clusters/n": sum_n / denom,
        "clusters/alpha": sum_alpha / denom,
    }

    return trace, post_means


# ------------------------------
# Helpers / pytest demo
# ------------------------------


def make_obs_from_y(y_obs):
    obs = ChoiceMap.empty()
    obs = obs.at["y"].set(y_obs)
    return obs


def test_main():
    key = jax.random.PRNGKey(123)
    xs = jnp.linspace(1.0, 100.0, 1000)
    alpha = jnp.ones(3)
    args = (xs, alpha, 100.0, 5.0, 4.0, 2.0, 0.05)

    key, sub = jax.random.split(key)
    tr_true = hill_mixture_model.simulate(sub, args)
    ch_true = tr_true.get_choices()
    print("[truth] weights:", ch_true["weights"])
    print("[truth] Kd:", ch_true["clusters/Kd"])
    print("[truth] n :", ch_true["clusters/n"])
    print("[truth] alpha :", ch_true["clusters/alpha"])

    y_obs = ch_true["y"]
    obs = make_obs_from_y(y_obs)

    key, sub = jax.random.split(key)
    tr_post, post_means = run_inference(
        sub,
        hill_mixture_model,
        args,
        obs,
        n_rounds=10000,
        mh_step_kd=0.03,
        mh_step_n=0.03,
        burn_in=1000,
        thin=5,
    )

    ch = tr_post.get_choices()
    print("[posterior mean] weights:", post_means["weights"])
    print("[posterior mean] Kd     :", post_means["clusters/Kd"])
    print("[posterior mean] n      :", post_means["clusters/n"])
    print("[posterior mean] alpha  :", post_means["clusters/alpha"])
    print("z head:", ch["y_mix", "mixture_component"][:10])

    assert jnp.all(ch["weights"] >= 0) and jnp.isclose(
        jnp.sum(ch["weights"]), 1.0, atol=1e-4
    )
