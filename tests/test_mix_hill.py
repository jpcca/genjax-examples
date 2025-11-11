"""
GenJAX-based Mixture of Hill functions with scalable MCMC (K=3 固定・修正版)
- genjax を使用（mix は不使用）
- 観測/割当は配列1アドレス（"y", "z"）でベクトル化
- weights は Dirichlet 共役 Gibbs
- (Kd, n) は log-space RW-MH（Gamma 事前）
- z は collapsed Gibbs
- K=3 固定。boolean マスク抽出を禁止し、重み付き総和に変更（JIT適合）
"""

from typing import Tuple, Dict, NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

from genjax import gen, Const
from genjax import normal, gamma, dirichlet, categorical

import matplotlib.pyplot as plt


# =====================================================
# Utilities
# =====================================================


def hill(
    x: jnp.ndarray, kd: jnp.ndarray, n: jnp.ndarray, eps: float = 1e-6
) -> jnp.ndarray:
    x = jnp.maximum(x, eps)
    return 1.0 / (1.0 + (kd / x) ** n)


def _safe_log(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return jnp.log(jnp.maximum(x, eps))


def log_normal_pdf(y, mu, sigma):
    return -0.5 * (
        jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(sigma) + ((y - mu) / sigma) ** 2
    )


# =====================================================
# Generative model (K=3 固定)
# =====================================================


@gen
def mix_hill(
    xs: jnp.ndarray,
    alpha: jnp.ndarray,  # (3,)
    shape_kd: float = 100.0,
    rate_kd: float = 5.0,
    shape_n: float = 4.0,
    rate_n: float = 2.0,
    sigma: float = 0.02,
):
    xs = jnp.asarray(xs)
    N = xs.shape[0]  # 静的整数

    # weights
    weights = dirichlet(alpha) @ "weights"  # (3,)

    # クラスタ・パラメータ
    Kd = gamma(shape_kd, rate_kd, sample_shape=Const((3,))) @ ("clusters", "Kd")  # (3,)
    n = gamma(shape_n, rate_n, sample_shape=Const((3,))) @ ("clusters", "n")  # (3,)

    # 割当 z をベクトル一括
    z = categorical(weights, sample_shape=Const((N,))) @ "z"  # (N,)

    # 各クラスタの平均を (N,3) で構築
    xs_safe = jnp.maximum(xs, 1e-6)
    mu_all = jnp.stack(
        [
            1.0 / (1.0 + (Kd[0] / xs_safe) ** n[0]),
            1.0 / (1.0 + (Kd[1] / xs_safe) ** n[1]),
            1.0 / (1.0 + (Kd[2] / xs_safe) ** n[2]),
        ],
        axis=1,
    )  # (N,3)

    # z でインデックス選択して mu_i を作る
    idx = jnp.arange(N)
    mu = mu_all[idx, z]  # (N,)

    # y をベクトル一括サンプル
    y = normal(mu, sigma) @ "y"  # (N,)
    return y


# =====================================================
# Collapsed Gibbs for z & Gibbs for weights
# =====================================================


@jit
def sample_z(
    key: jax.Array,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    weights: jnp.ndarray,  # (3,)
    Kd: jnp.ndarray,  # (3,)
    n: jnp.ndarray,  # (3,)
    sigma: float,
) -> Tuple[jax.Array, jnp.ndarray]:
    def logit_row(xi, yi):
        mu_k = hill(jnp.full_like(Kd, xi), Kd, n)  # (3,)
        ll_k = log_normal_pdf(yi, mu_k, sigma)  # (3,)
        return _safe_log(weights) + ll_k

    logits = vmap(logit_row)(xs, ys)  # (N, 3)
    key, sub = jax.random.split(key)
    z = jax.random.categorical(sub, logits, axis=-1)
    return key, z


@jit
def sample_weights(
    key: jax.Array, alpha: jnp.ndarray, z: jnp.ndarray
) -> Tuple[jax.Array, jnp.ndarray]:
    # jit 互換: length を静的指定
    counts = jnp.bincount(z, length=3).astype(alpha.dtype)
    key, sub = jax.random.split(key)
    w = jax.random.dirichlet(sub, alpha + counts)
    return key, w


# =====================================================
# MH for (Kd, n) – log-space RW（boolean マスク抽出禁止）
# =====================================================


class MHConfig(NamedTuple):
    step_kd: float
    step_n: float


def _mh_one_param(key, current, step, logpost_fn):
    key, sub = jax.random.split(key)
    prop = current + step * jax.random.normal(sub, ())
    loga = jnp.minimum(0.0, logpost_fn(prop) - logpost_fn(current))
    key, sub = jax.random.split(key)
    accept = jnp.log(jax.random.uniform(sub, ())) < loga
    new = jnp.where(accept, prop, current)
    return key, new, accept


def _logpost_kd_factory(k: int, xs, ys, z, Kd_log, n_log, sigma, shape_kd, rate_kd):
    """
    以前: xs_k = xs[z==k] のような抽出 → JIT非対応
    現在: mask = 1[z==k] を掛けて全データ総和
    """
    mask = (z == k).astype(ys.dtype)  # (N,)

    def logpost(kd_log_prime):
        kd_prime = jnp.exp(kd_log_prime)
        n_k = jnp.exp(n_log[k])
        log_prior = (shape_kd - 1.0) * kd_log_prime - rate_kd * kd_prime

        mu_all = hill(xs, kd_prime, n_k)  # (N,)
        ll_all = log_normal_pdf(ys, mu_all, sigma)  # (N,)
        ll = jnp.sum(mask * ll_all)  # そのクラスタに属す点のみ加算
        return log_prior + ll

    return logpost


def _logpost_n_factory(k: int, xs, ys, z, Kd_log, n_log, sigma, shape_n, rate_n):
    mask = (z == k).astype(ys.dtype)  # (N,)

    def logpost(n_log_prime):
        n_prime = jnp.exp(n_log_prime)
        kd_k = jnp.exp(Kd_log[k])
        log_prior = (shape_n - 1.0) * n_log_prime - rate_n * n_prime

        mu_all = hill(xs, kd_k, n_prime)  # (N,)
        ll_all = log_normal_pdf(ys, mu_all, sigma)  # (N,)
        ll = jnp.sum(mask * ll_all)
        return log_prior + ll

    return logpost


@jit
def mh_update_params(
    key: jax.Array,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    z: jnp.ndarray,
    Kd: jnp.ndarray,
    n: jnp.ndarray,
    sigma: float,
    shape_kd: float,
    rate_kd: float,
    shape_n: float,
    rate_n: float,
    cfg: MHConfig,
) -> Tuple[jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Kd_log = jnp.log(Kd)
    n_log = jnp.log(n)

    def upd_k(k, carry):
        key, Kd_log_vec, n_log_vec, acc_kd_vec, acc_n_vec = carry

        key, new_kd_log, acc_kd = _mh_one_param(
            key,
            Kd_log_vec[k],
            cfg.step_kd,
            _logpost_kd_factory(k, xs, ys, z, Kd_log, n_log, sigma, shape_kd, rate_kd),
        )
        Kd_log_vec = Kd_log_vec.at[k].set(new_kd_log)

        key, new_n_log, acc_n = _mh_one_param(
            key,
            n_log_vec[k],
            cfg.step_n,
            _logpost_n_factory(k, xs, ys, z, Kd_log_vec, n_log, sigma, shape_n, rate_n),
        )
        n_log_vec = n_log_vec.at[k].set(new_n_log)

        acc_kd_vec = acc_kd_vec.at[k].add(acc_kd.astype(acc_kd_vec.dtype))
        acc_n_vec = acc_n_vec.at[k].add(acc_n.astype(acc_n_vec.dtype))
        return (key, Kd_log_vec, n_log_vec, acc_kd_vec, acc_n_vec)

    init = (key, Kd_log, n_log, jnp.zeros((3,)), jnp.zeros((3,)))
    key, Kd_log, n_log, acc_kd, acc_n = lax.fori_loop(0, 3, upd_k, init)
    return key, jnp.exp(Kd_log), jnp.exp(n_log), acc_kd, acc_n


# =====================================================
# Full MCMC driver
# =====================================================


class MCMCConfig(NamedTuple):
    num_iters: int = 2000
    burn_in: int = 1000
    thin: int = 10
    mh_step_kd: float = 0.05
    mh_step_n: float = 0.05


class MCMCState(NamedTuple):
    key: jax.Array
    weights: jnp.ndarray  # (3,)
    z: jnp.ndarray  # (N,)
    Kd: jnp.ndarray  # (3,)
    n: jnp.ndarray  # (3,)


@jit
def mcmc_step(
    state: MCMCState,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    alpha: jnp.ndarray,
    shape_kd: float,
    rate_kd: float,
    shape_n: float,
    rate_n: float,
    sigma: float,
    mh_cfg: MHConfig,
) -> Tuple[MCMCState, Tuple[jnp.ndarray, jnp.ndarray]]:
    key = state.key

    # z | rest (collapsed Gibbs)
    key, z = sample_z(key, xs, ys, state.weights, state.Kd, state.n, sigma)

    # w | z (Gibbs)
    key, w = sample_weights(key, alpha, z)

    # (Kd,n) | rest (MH)
    key, Kd_new, n_new, acc_kd, acc_n = mh_update_params(
        key,
        xs,
        ys,
        z,
        state.Kd,
        state.n,
        sigma,
        shape_kd,
        rate_kd,
        shape_n,
        rate_n,
        mh_cfg,
    )

    new_state = MCMCState(key, w, z, Kd_new, n_new)
    return new_state, (acc_kd, acc_n)


def run_mcmc(
    key: jax.Array,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    alpha: jnp.ndarray,  # 形状 (3,)
    shape_kd: float = 100.0,
    rate_kd: float = 5.0,
    shape_n: float = 4.0,
    rate_n: float = 2.0,
    sigma: float = 0.02,
    cfg: MCMCConfig = MCMCConfig(),
) -> Dict[str, jnp.ndarray]:
    xs = jnp.asarray(xs)
    ys = jnp.asarray(ys)

    # 初期化: 事前から一度だけサンプル
    key, sub = jax.random.split(key)
    tr = mix_hill.simulate(sub, (xs, alpha, shape_kd, rate_kd, shape_n, rate_n, sigma))
    ch = tr.get_choices()
    w0 = ch["weights"]  # (3,)
    Kd0 = ch[("clusters", "Kd")]  # (3,)
    n0 = ch[("clusters", "n")]  # (3,)

    key, z0 = sample_z(key, xs, ys, w0, Kd0, n0, sigma)
    state = MCMCState(key, w0, z0, Kd0, n0)

    mh_cfg = MHConfig(cfg.mh_step_kd, cfg.mh_step_n)

    def one_step(carry, _):
        st, t, sum_w, sum_kd, sum_n, acc_kd_sum, acc_n_sum, kept = carry
        st, (acc_kd, acc_n) = mcmc_step(
            st, xs, ys, alpha, shape_kd, rate_kd, shape_n, rate_n, sigma, mh_cfg
        )
        t = t + 1

        def do_keep(args):
            st, sum_w, sum_kd, sum_n, acc_kd_sum, acc_n_sum, kept = args
            return (
                st,
                sum_w + st.weights,
                sum_kd + st.Kd,
                sum_n + st.n,
                acc_kd_sum + acc_kd,
                acc_n_sum + acc_n,
                kept + 1,
            )

        def no_keep(args):
            return args

        keep_cond = jnp.logical_and(t > cfg.burn_in, (t - cfg.burn_in) % cfg.thin == 0)
        sum_w, sum_kd, sum_n, acc_kd_sum, acc_n_sum, kept = lax.cond(
            keep_cond,
            do_keep,
            no_keep,
            operand=(st, sum_w, sum_kd, sum_n, acc_kd_sum, acc_n_sum, kept),
        )[1:]

        return (st, t, sum_w, sum_kd, sum_n, acc_kd_sum, acc_n_sum, kept), None

    init_carry = (
        state,
        jnp.array(0, dtype=jnp.int32),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.array(0, dtype=jnp.int32),
    )

    (st, t, sum_w, sum_kd, sum_n, acc_kd_sum, acc_n_sum, kept), _ = lax.scan(
        lambda c, i: one_step(c, i), init_carry, xs=jnp.arange(cfg.num_iters)
    )

    kept = jnp.maximum(kept, 1)
    post_w = sum_w / kept
    post_kd = sum_kd / kept
    post_n = sum_n / kept
    acc_kd_rate = acc_kd_sum / jnp.maximum(cfg.num_iters, 1)
    acc_n_rate = acc_n_sum / jnp.maximum(cfg.num_iters, 1)

    return dict(
        post_w=post_w,
        post_kd=post_kd,
        post_n=post_n,
        acc_kd=acc_kd_rate,
        acc_n=acc_n_rate,
        last_state=st,
    )


# =====================================================
# Demo helpers / pytest
# =====================================================


def simulate_truth_and_data(
    key: jax.Array,
    N: int = 500,
    alpha=(1.0, 1.0, 1.0),
    shape_kd=100.0,
    rate_kd=5.0,
    shape_n=4.0,
    rate_n=2.0,
    sigma=0.02,
):
    xs = jnp.linspace(1.0, 100.0, N)
    alpha = jnp.asarray(alpha)
    key, sub = jax.random.split(key)

    tr = mix_hill.simulate(sub, (xs, alpha, shape_kd, rate_kd, shape_n, rate_n, sigma))
    ys = tr.get_retval()
    ch = tr.get_choices()
    truth = {
        "weights": ch["weights"],  # (3,)
        "Kd": ch[("clusters", "Kd")],  # (3,)
        "n": ch[("clusters", "n")],  # (3,)
        "z": ch["z"],  # (N,)
    }
    return key, xs, ys, truth


def test_main():
    key = jax.random.PRNGKey(314159)
    key, xs, ys, truth = simulate_truth_and_data(key, N=1000)
    print("[truth] weights:", truth["weights"])
    print("[truth] Kd:", truth["Kd"])
    print("[truth] n :", truth["n"])

    alpha = jnp.ones(3)
    out = run_mcmc(
        key,
        xs,
        ys,
        alpha=alpha,
        cfg=MCMCConfig(
            num_iters=2000, burn_in=1000, thin=10, mh_step_kd=0.03, mh_step_n=0.03
        ),
        # Wider priors to increase posterior spread
        shape_kd=4.0,
        rate_kd=0.2,
        shape_n=1.0,
        rate_n=0.5,
    )
    print("[posterior mean] weights:", out["post_w"])
    # Note: no post_alpha in output; weights shown above
    print("[posterior mean] beta(Kd):", out["post_kd"])
    print("[posterior mean] gamma(n):", out["post_n"])
    if "acc_alpha" in out:
        print(
            "[accept rates] alpha, kd, n:",
            out["acc_alpha"],
            out["acc_kd"],
            out["acc_n"],
        )
    else:
        print("[accept rates] kd, n    :", out["acc_kd"], out["acc_n"])

    # Plot: data (orange), 3 posterior-mean Hill curves (black),
    # and for the last 500 samples, each component's Hill curve in gray.

    # Collect last 500 samples by advancing from the last state
    last_kds = []
    last_ns = []
    st = out["last_state"]
    mh_cfg = MHConfig(0.03, 0.03)
    for _ in range(500):
        st, _ = mcmc_step(
            st,
            xs,
            ys,
            alpha,
            100.0,
            5.0,
            4.0,
            2.0,
            0.02,
            mh_cfg,
        )
        last_kds.append(st.Kd)
        last_ns.append(st.n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, s=10, color="orange", edgecolors="none", label="data")

    # Last 500 sample component curves (gray)
    for i in range(len(last_kds)):
        for k in range(3):
            mu_i_k = hill(xs, last_kds[i][k], last_ns[i][k])
            ax.plot(xs, mu_i_k, color="#888888", alpha=0.3, linewidth=1)

    # Posterior-mean component curves (black)
    for k in range(3):
        # Use posterior means correctly: hill(x, kd, n)
        mu_post_k = hill(xs, out["post_kd"][k], out["post_n"][k])
        ax.plot(
            xs,
            mu_post_k,
            color="black",
            linewidth=2,
            label="posterior mean" if k == 0 else None,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Mixture of Hill Functions: Data and Posterior Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig("mix_hill_plot.png", dpi=150)
