# -*- coding: utf-8 -*-
"""
GenJAX-only inference combinators for Mixture of Hill functions (K=3, traceをproposalに渡さない版)
"""

import jax
import jax.numpy as jnp

import genjax
from genjax import gen, normal, gamma, dirichlet, categorical  # type: ignore
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
# Generative model (vectorized addresses)
# ------------------------------
@gen
def mix_hill_vec(
    xs,
    alpha,
    shape_kd=100.0,
    rate_kd=5.0,
    shape_n=4.0,
    rate_n=2.0,
    sigma=0.02,
):
    xs = xs
    N = xs.shape[0]
    K = 3

    w = dirichlet(alpha) @ "weights"  # (K,)
    # Use flat string addresses to avoid mixed key types in pytrees
    Kd = gamma(shape_kd, rate_kd, sample_shape=Const((K,))) @ "clusters/Kd"  # (K,)
    n = gamma(shape_n, rate_n, sample_shape=Const((K,))) @ "clusters/n"  # (K,)

    z = categorical(w, sample_shape=Const((N,))) @ "z"  # (N,)

    xs_safe = jnp.maximum(xs, 1e-6)
    mu_all = jnp.stack(
        [
            1.0 / (1.0 + (Kd[0] / xs_safe) ** n[0]),
            1.0 / (1.0 + (Kd[1] / xs_safe) ** n[1]),
            1.0 / (1.0 + (Kd[2] / xs_safe) ** n[2]),
        ],
        axis=1,
    )  # (N, K)
    mu = mu_all[jnp.arange(N), z]  # (N,)

    y = normal(mu, sigma) @ "y"
    return y


# ------------------------------
# Proposals（traceは受け取らない）
# ------------------------------


# MH proposal for (Kd, n): log-space symmetric RW
@gen
def rw_globals_prop(Kd_cur, n_cur, step_kd: float, step_n: float):
    logKd = jnp.log(Kd_cur)
    logn = jnp.log(n_cur)

    eps_kd = (
        normal.vmap()(jnp.zeros_like(logKd), step_kd * jnp.ones_like(logKd)) @ "eps_kd"
    )
    eps_n = normal.vmap()(jnp.zeros_like(logn), step_n * jnp.ones_like(logn)) @ "eps_n"

    Kd_prop = jnp.exp(logKd + eps_kd)
    n_prop = jnp.exp(logn + eps_n)
    return (Kd_prop, n_prop)


# Gibbs proposal for weights: Dirichlet(alpha + counts(z))
@gen
def gibbs_weights_prop(alpha, z_vec):
    K = alpha.shape[0]
    counts = jnp.bincount(z_vec, length=K)
    post_alpha = alpha + counts
    w_new = dirichlet(post_alpha) @ "w_new"
    return w_new


# Gibbs proposal for z (collapsed) 一括
@gen
def gibbs_z_prop(xs, y_vec, weights, Kd, n, sigma):
    def logits_row(xi, yi):
        mu_k = hill(jnp.full((3,), xi), Kd, n)  # (3,)
        ll_k = -0.5 * (jnp.log(2.0 * jnp.pi * sigma**2) + ((yi - mu_k) / sigma) ** 2)
        return jnp.log(weights + 1e-20) + ll_k  # (3,)

    logits = jax.vmap(logits_row)(xs, y_vec)  # (N, 3)
    # ★ 行ごとの独立カテゴリカルにする（バッチ化）
    z = categorical.vmap()(logits) @ "z"  # (N,)
    return z


# ------------------------------
# Single-step kernels using propose/assess + model.update
# ------------------------------


def mh_step_globals(key, trace, model, step_kd: float, step_n: float):
    """(Kd, n) を対称 RW 提案で MH。提案は simulate で取り、assess は使わない。"""
    argdiffs = Diff.no_change(trace.get_args())
    chm_cur = trace.get_choices()
    Kd_cur = chm_cur["clusters/Kd"]
    n_cur = chm_cur["clusters/n"]

    # 対称 RW 提案（log 空間）
    key, sub = jax.random.split(key)
    Kd_prop, n_prop = rw_globals_prop.simulate(
        sub, (Kd_cur, n_cur, step_kd, step_n)
    ).get_retval()

    # 提案をモデルに適用
    chm = ChoiceMap.empty()
    chm = chm.at["clusters/Kd"].set(Kd_prop)
    chm = chm.at["clusters/n"].set(n_prop)

    key, sub = jax.random.split(key)
    new_trace, model_logw, _, _ = model.update(sub, trace, chm, argdiffs)

    # 対称提案なので fwd/bwd は 0、受容確率は model_logw のみで決まる
    key, sub = jax.random.split(key)
    accept = jnp.log(jax.random.uniform(sub)) < model_logw
    # Use lax.cond to select traces under JIT/scan
    out_trace = jax.lax.cond(accept, lambda _: new_trace, lambda _: trace, operand=None)
    return key, out_trace, accept, model_logw


def gibbs_step_weights(key, trace, model, alpha):
    """weights | z はディリクレ共役。simulate でサンプルして確定 update。"""
    argdiffs = Diff.no_change(trace.get_args())
    z_vec = trace.get_choices()["z"]

    key, sub = jax.random.split(key)
    w_new = gibbs_weights_prop.simulate(sub, (alpha, z_vec)).get_retval()

    chm = ChoiceMap.empty().at["weights"].set(w_new)
    key, sub = jax.random.split(key)
    new_trace, _, _, _ = model.update(sub, trace, chm, argdiffs)
    return key, new_trace


def gibbs_step_z(key, trace, model, sigma):
    argdiffs = Diff.no_change(trace.get_args())
    ch = trace.get_choices()
    xs = trace.get_args()[0]
    y_vec = ch["y"]
    w = ch["weights"]
    Kd = ch["clusters/Kd"]
    n = ch["clusters/n"]

    key, sub = jax.random.split(key)
    z_new = gibbs_z_prop.simulate(sub, (xs, y_vec, w, Kd, n, sigma)).get_retval()
    # dtype/shape を明示（int32, (N,)）
    z_new = z_new.astype(jnp.int32).reshape(y_vec.shape)

    chm = ChoiceMap.empty().at["z"].set(z_new)
    key, sub = jax.random.split(key)
    new_trace, _, _, _ = model.update(sub, trace, chm, argdiffs)
    return key, new_trace


# ------------------------------
# Full inference loop
# ------------------------------


def run_inference_genjax_only(
    key,
    model,
    model_args,
    observations,
    n_rounds=200,
    mh_step_kd=0.03,
    mh_step_n=0.03,
    do_gibbs_z=True,
    do_gibbs_w=True,
):
    key, sub = jax.random.split(key)
    trace, _ = model.importance(sub, observations, model_args)
    alpha = model_args[1]
    sigma = model_args[-1]

    # JIT-friendly loop with lax.scan; guard Gibbs steps via lax.cond
    def _step(carry, _):
        key_s, trace_s = carry
        key_s, trace_s, _, _ = mh_step_globals(
            key_s, trace_s, model, mh_step_kd, mh_step_n
        )
        key_s, trace_s = jax.lax.cond(
            do_gibbs_z,
            lambda kt: gibbs_step_z(kt[0], kt[1], model, sigma),
            lambda kt: kt,
            (key_s, trace_s),
        )
        key_s, trace_s = jax.lax.cond(
            do_gibbs_w,
            lambda kt: gibbs_step_weights(kt[0], kt[1], model, alpha),
            lambda kt: kt,
            (key_s, trace_s),
        )
        return (key_s, trace_s), None

    (key, trace), _ = jax.lax.scan(_step, (key, trace), xs=None, length=n_rounds)
    return trace


# ------------------------------
# Helpers / pytest demo
# ------------------------------


def make_obs_from_y(y_obs):
    obs = ChoiceMap.empty()
    obs = obs.at["y"].set(y_obs)
    return obs


def test_main():
    key = jax.random.PRNGKey(314159)
    xs = jnp.linspace(1.0, 100.0, 1000)
    alpha = jnp.ones(3)
    args = (xs, alpha, 100.0, 5.0, 4.0, 2.0, 0.05)

    key, sub = jax.random.split(key)
    tr_true = mix_hill_vec.simulate(sub, args)
    ch_true = tr_true.get_choices()
    print("[truth] weights:", ch_true["weights"])
    print("[truth] Kd:", ch_true["clusters/Kd"])
    print("[truth] n :", ch_true["clusters/n"])

    y_obs = ch_true["y"]
    obs = make_obs_from_y(y_obs)

    key, sub = jax.random.split(key)
    tr_post = run_inference_genjax_only(
        sub,
        mix_hill_vec,
        args,
        obs,
        n_rounds=2000,
        mh_step_kd=0.03,
        mh_step_n=0.03,
        do_gibbs_z=True,
        do_gibbs_w=True,
    )

    ch = tr_post.get_choices()
    print("[posterior] weights:", ch["weights"])
    print("[posterior] Kd     :", ch["clusters/Kd"])
    print("[posterior] n      :", ch["clusters/n"])
    print("z head:", ch["z"][:10])

    assert jnp.all(ch["weights"] >= 0) and jnp.isclose(
        jnp.sum(ch["weights"]), 1.0, atol=1e-4
    )
