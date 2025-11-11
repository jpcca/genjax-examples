"""
Three-parameter Mixture of Hill functions (adds amplitude alpha) with MCMC

What’s new vs tests/test_mix_hill.py:
- Each component k has an amplitude alpha[k] that scales the Hill curve: mu = alpha[k] * 1/(1+(Kd/x)^n)
- We infer alpha alongside Kd and n via log-space RW-MH (Gamma priors)
- Demo applying the model to tests/data/mmm.csv grouped by
  (ORGANISATION_SUBVERTICAL, TERRITORY_NAME, MARKETING_CHANNEL),
  with SPEND as x and CLICKS as y. We ignore IMPRESSIONS.

Notes:
- For clarity, the Dirichlet prior hyperparameter for mixture weights is named dir_alpha
  to avoid confusion with the amplitude parameters alpha.
- Default K=2 components; code is written JIT‑friendly (no boolean indexing; use masks).
"""

from typing import Tuple, Dict, NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

from genjax import gen, Const
from genjax import normal, gamma, dirichlet, categorical

import pandas as pd
import matplotlib.pyplot as plt

# Number of mixture components (change here if needed)
K = 2

# Dirichlet concentration for mixture weights (symmetric)
# Larger values (e.g., 5.0 or 10.0) encourage more balanced component usage
DIRICHLET_CONC = 10.0


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
# Generative model (K fixed) with amplitude alpha per component
# =====================================================


@gen
def mix_hill_3param(
    xs: jnp.ndarray,
    dir_alpha: jnp.ndarray,  # (K,) Dirichlet prior for weights
    # Priors for cluster parameters
    shape_kd: float = 100.0,
    rate_kd: float = 5.0,
    shape_n: float = 4.0,
    rate_n: float = 2.0,
    shape_a: float = 2.0,
    rate_a: float = 0.1,
    sigma: float = 1.0,
):
    xs = jnp.asarray(xs)
    N = xs.shape[0]

    # weights
    weights = dirichlet(dir_alpha) @ "weights"  # (K,)

    # Cluster parameters
    Kd = gamma(shape_kd, rate_kd, sample_shape=Const((K,))) @ ("clusters", "Kd")  # (K,)
    n = gamma(shape_n, rate_n, sample_shape=Const((K,))) @ ("clusters", "n")  # (K,)
    A = gamma(shape_a, rate_a, sample_shape=Const((K,))) @ ("clusters", "alpha")  # (K,)

    # Assignments
    z = categorical(weights, sample_shape=Const((N,))) @ "z"  # (N,)

    # Means for all components (N,K)
    xs_safe = jnp.maximum(xs, 1e-6)
    # Vectorized construction for general K
    base = 1.0 / (1.0 + (Kd[None, :] / xs_safe[:, None]) ** n[None, :])  # (N,K)
    mu_all = base * A  # (N,K)

    idx = jnp.arange(N)
    mu = mu_all[idx, z]

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
    weights: jnp.ndarray,  # (K,)
    Kd: jnp.ndarray,  # (K,)
    n: jnp.ndarray,  # (K,)
    A: jnp.ndarray,  # (K,)
    sigma: float,
) -> Tuple[jax.Array, jnp.ndarray]:
    def logit_row(xi, yi):
        base = hill(jnp.full_like(Kd, xi), Kd, n)  # (K,)
        mu_k = A * base  # (K,)
        ll_k = log_normal_pdf(yi, mu_k, sigma)  # (K,)
        return _safe_log(weights) + ll_k

    logits = vmap(logit_row)(xs, ys)  # (N, K)
    key, sub = jax.random.split(key)
    z = jax.random.categorical(sub, logits, axis=-1)
    return key, z


@jit
def sample_weights(
    key: jax.Array, dir_alpha: jnp.ndarray, z: jnp.ndarray
) -> Tuple[jax.Array, jnp.ndarray]:
    counts = jnp.bincount(z, length=K).astype(dir_alpha.dtype)
    key, sub = jax.random.split(key)
    w = jax.random.dirichlet(sub, dir_alpha + counts)
    return key, w


# =====================================================
# MH for (Kd, n, A) – log-space RW（mask sum; avoids boolean indexing）
# =====================================================


class MHConfig(NamedTuple):
    step_kd: float
    step_n: float
    step_a: float


def _mh_one_param(key, current, step, logpost_fn):
    key, sub = jax.random.split(key)
    prop = current + step * jax.random.normal(sub, ())
    loga = jnp.minimum(0.0, logpost_fn(prop) - logpost_fn(current))
    key, sub = jax.random.split(key)
    accept = jnp.log(jax.random.uniform(sub, ())) < loga
    new = jnp.where(accept, prop, current)
    return key, new, accept


def _logpost_kd_factory(
    k: int, xs, ys, z, Kd_log, n_log, A_log, sigma, shape_kd, rate_kd
):
    mask = (z == k).astype(ys.dtype)

    def logpost(kd_log_prime):
        kd_prime = jnp.exp(kd_log_prime)
        n_k = jnp.exp(n_log[k])
        a_k = jnp.exp(A_log[k])
        log_prior = (shape_kd - 1.0) * kd_log_prime - rate_kd * kd_prime

        base = hill(xs, kd_prime, n_k)
        mu_all = a_k * base
        ll_all = log_normal_pdf(ys, mu_all, sigma)
        ll = jnp.sum(mask * ll_all)
        return log_prior + ll

    return logpost


def _logpost_n_factory(k: int, xs, ys, z, Kd_log, n_log, A_log, sigma, shape_n, rate_n):
    mask = (z == k).astype(ys.dtype)

    def logpost(n_log_prime):
        n_prime = jnp.exp(n_log_prime)
        kd_k = jnp.exp(Kd_log[k])
        a_k = jnp.exp(A_log[k])
        log_prior = (shape_n - 1.0) * n_log_prime - rate_n * n_prime

        base = hill(xs, kd_k, n_prime)
        mu_all = a_k * base
        ll_all = log_normal_pdf(ys, mu_all, sigma)
        ll = jnp.sum(mask * ll_all)
        return log_prior + ll

    return logpost


def _logpost_a_factory(k: int, xs, ys, z, Kd_log, n_log, A_log, sigma, shape_a, rate_a):
    mask = (z == k).astype(ys.dtype)

    def logpost(a_log_prime):
        a_prime = jnp.exp(a_log_prime)
        kd_k = jnp.exp(Kd_log[k])
        n_k = jnp.exp(n_log[k])
        log_prior = (shape_a - 1.0) * a_log_prime - rate_a * a_prime

        base = hill(xs, kd_k, n_k)
        mu_all = a_prime * base
        ll_all = log_normal_pdf(ys, mu_all, sigma)
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
    A: jnp.ndarray,
    sigma: float,
    shape_kd: float,
    rate_kd: float,
    shape_n: float,
    rate_n: float,
    shape_a: float,
    rate_a: float,
    cfg: MHConfig,
) -> Tuple[jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Kd_log = jnp.log(Kd)
    n_log = jnp.log(n)
    A_log = jnp.log(A)

    def upd_k(k, carry):
        key, Kd_log_vec, n_log_vec, A_log_vec, acc_kd_vec, acc_n_vec, acc_a_vec = carry

        key, new_kd_log, acc_kd = _mh_one_param(
            key,
            Kd_log_vec[k],
            cfg.step_kd,
            _logpost_kd_factory(
                k, xs, ys, z, Kd_log_vec, n_log_vec, A_log_vec, sigma, shape_kd, rate_kd
            ),
        )
        Kd_log_vec = Kd_log_vec.at[k].set(new_kd_log)

        key, new_n_log, acc_n = _mh_one_param(
            key,
            n_log_vec[k],
            cfg.step_n,
            _logpost_n_factory(
                k, xs, ys, z, Kd_log_vec, n_log_vec, A_log_vec, sigma, shape_n, rate_n
            ),
        )
        n_log_vec = n_log_vec.at[k].set(new_n_log)

        key, new_a_log, acc_a = _mh_one_param(
            key,
            A_log_vec[k],
            cfg.step_a,
            _logpost_a_factory(
                k, xs, ys, z, Kd_log_vec, n_log_vec, A_log_vec, sigma, shape_a, rate_a
            ),
        )
        A_log_vec = A_log_vec.at[k].set(new_a_log)

        acc_kd_vec = acc_kd_vec.at[k].add(acc_kd.astype(acc_kd_vec.dtype))
        acc_n_vec = acc_n_vec.at[k].add(acc_n.astype(acc_n_vec.dtype))
        acc_a_vec = acc_a_vec.at[k].add(acc_a.astype(acc_a_vec.dtype))
        return (key, Kd_log_vec, n_log_vec, A_log_vec, acc_kd_vec, acc_n_vec, acc_a_vec)

    init = (
        key,
        Kd_log,
        n_log,
        A_log,
        jnp.zeros((K,)),
        jnp.zeros((K,)),
        jnp.zeros((K,)),
    )
    key, Kd_log, n_log, A_log, acc_kd, acc_n, acc_a = lax.fori_loop(0, K, upd_k, init)
    return key, jnp.exp(Kd_log), jnp.exp(n_log), jnp.exp(A_log), acc_kd, acc_n, acc_a


# =====================================================
# Full MCMC driver
# =====================================================


class MCMCConfig(NamedTuple):
    num_iters: int = 1000
    burn_in: int = 500
    thin: int = 10
    mh_step_kd: float = 0.05
    mh_step_n: float = 0.05
    mh_step_a: float = 0.05


class MCMCState(NamedTuple):
    key: jax.Array
    weights: jnp.ndarray  # (K,)
    z: jnp.ndarray  # (N,)
    Kd: jnp.ndarray  # (K,)
    n: jnp.ndarray  # (K,)
    A: jnp.ndarray  # (K,) amplitude


@jit
def mcmc_step(
    state: MCMCState,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    dir_alpha: jnp.ndarray,
    shape_kd: float,
    rate_kd: float,
    shape_n: float,
    rate_n: float,
    shape_a: float,
    rate_a: float,
    sigma: float,
    mh_cfg: MHConfig,
) -> Tuple[MCMCState, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    key = state.key

    # z | rest (collapsed Gibbs)
    key, z = sample_z(key, xs, ys, state.weights, state.Kd, state.n, state.A, sigma)

    # w | z (Gibbs)
    key, w = sample_weights(key, dir_alpha, z)

    # (Kd,n,A) | rest (MH)
    key, Kd_new, n_new, A_new, acc_kd, acc_n, acc_a = mh_update_params(
        key,
        xs,
        ys,
        z,
        state.Kd,
        state.n,
        state.A,
        sigma,
        shape_kd,
        rate_kd,
        shape_n,
        rate_n,
        shape_a,
        rate_a,
        mh_cfg,
    )

    new_state = MCMCState(key, w, z, Kd_new, n_new, A_new)
    return new_state, (acc_kd, acc_n, acc_a)


def run_mcmc(
    key: jax.Array,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    dir_alpha: jnp.ndarray,  # shape (K,)
    shape_kd: float = 100.0,
    rate_kd: float = 5.0,
    shape_n: float = 4.0,
    rate_n: float = 2.0,
    shape_a: float = 2.0,
    rate_a: float = 0.1,
    sigma: float = 1.0,
    cfg: MCMCConfig = MCMCConfig(),
) -> Dict[str, jnp.ndarray]:
    xs = jnp.asarray(xs)
    ys = jnp.asarray(ys)

    # Initialize from prior once
    key, sub = jax.random.split(key)
    tr = mix_hill_3param.simulate(
        sub, (xs, dir_alpha, shape_kd, rate_kd, shape_n, rate_n, shape_a, rate_a, sigma)
    )
    ch = tr.get_choices()
    w0 = ch["weights"]  # (K,)
    Kd0 = ch[("clusters", "Kd")]  # (K,)
    n0 = ch[("clusters", "n")]  # (K,)
    A0 = ch[("clusters", "alpha")]  # (K,)

    key, z0 = sample_z(key, xs, ys, w0, Kd0, n0, A0, sigma)
    state = MCMCState(key, w0, z0, Kd0, n0, A0)

    mh_cfg = MHConfig(cfg.mh_step_kd, cfg.mh_step_n, cfg.mh_step_a)

    def one_step(carry, _):
        st, t, sum_w, sum_kd, sum_n, sum_a, acc_kd_sum, acc_n_sum, acc_a_sum, kept = (
            carry
        )
        st, (acc_kd, acc_n, acc_a) = mcmc_step(
            st,
            xs,
            ys,
            dir_alpha,
            shape_kd,
            rate_kd,
            shape_n,
            rate_n,
            shape_a,
            rate_a,
            sigma,
            mh_cfg,
        )
        t = t + 1
        do_keep = (t > cfg.burn_in) & ((t - cfg.burn_in) % cfg.thin == 0)
        sum_w = jnp.where(do_keep, sum_w + st.weights, sum_w)
        sum_kd = jnp.where(do_keep, sum_kd + st.Kd, sum_kd)
        sum_n = jnp.where(do_keep, sum_n + st.n, sum_n)
        sum_a = jnp.where(do_keep, sum_a + st.A, sum_a)
        acc_kd_sum = acc_kd_sum + acc_kd
        acc_n_sum = acc_n_sum + acc_n
        acc_a_sum = acc_a_sum + acc_a
        kept = kept + do_keep.astype(kept.dtype)
        return (
            st,
            t,
            sum_w,
            sum_kd,
            sum_n,
            sum_a,
            acc_kd_sum,
            acc_n_sum,
            acc_a_sum,
            kept,
        ), None

    st0 = state
    carry0 = (
        st0,
        jnp.array(0),
        jnp.zeros((K,)),
        jnp.zeros((K,)),
        jnp.zeros((K,)),
        jnp.zeros((K,)),
        jnp.zeros((K,)),
        jnp.zeros((K,)),
        jnp.zeros((K,)),
        jnp.array(0),
    )

    (st, t, sum_w, sum_kd, sum_n, sum_a, acc_kd_sum, acc_n_sum, acc_a_sum, kept), _ = (
        lax.scan(one_step, carry0, xs=None, length=cfg.num_iters)
    )
    kept = jnp.maximum(kept, 1)
    post_w = sum_w / kept
    post_kd = sum_kd / kept
    post_n = sum_n / kept
    post_a = sum_a / kept
    acc_kd_rate = acc_kd_sum / cfg.num_iters
    acc_n_rate = acc_n_sum / cfg.num_iters
    acc_a_rate = acc_a_sum / cfg.num_iters

    return dict(
        post_w=post_w,
        post_kd=post_kd,
        post_n=post_n,
        post_a=post_a,
        acc_kd=acc_kd_rate,
        acc_n=acc_n_rate,
        acc_a=acc_a_rate,
        last_state=st,
    )


# =====================================================
# Demo: Apply to tests/data/mmm.csv (single group for speed)
# =====================================================


def _plot_group(xs, ys, post_w, post_kd, post_n, post_a, st_last, title, out_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, s=10, color="orange", edgecolors="none", label="data")

    # Sort x for line plots to avoid criss-crossing that looks like multiple lines
    order = jnp.argsort(xs)
    xs_line = xs[order]

    # # Draw last 200 sample curves (advance from last state)
    # last_kds = []
    # last_ns = []
    # last_as = []
    # mh_cfg = MHConfig(0.03, 0.03, 0.03)
    # key = st_last.key
    # st = st_last
    # for _ in range(200):
    #     st, _ = mcmc_step(
    #         st,
    #         xs,
    #         ys,
    #         jnp.ones(3),  # dirichlet hyper; not used here
    #         100.0,
    #         5.0,
    #         4.0,
    #         2.0,
    #         2.0,
    #         0.1,
    #         jnp.std(ys) * 0.5 + 1e-6,
    #         mh_cfg,
    #     )
    #     last_kds.append(st.Kd)
    #     last_ns.append(st.n)
    #     last_as.append(st.A)

    # for i in range(len(last_kds)):
    #     for k in range(K):
    #         mu_i_k = last_as[i][k] * hill(xs, last_kds[i][k], last_ns[i][k])
    #         ax.plot(xs, mu_i_k, color="#888", alpha=0.2, linewidth=1)

    # Plot posterior-mean component curves with opacity proportional to weight
    # Draw lighter first, heavier last to make weight effect visible
    order_k = list(jnp.argsort(post_w))  # ascending
    for idx, k in enumerate(order_k):
        w_k = float(post_w[int(k)])
        mu_post_k = post_a[int(k)] * hill(xs_line, post_kd[int(k)], post_n[int(k)])
        alpha_k = float(0.2 + 0.8 * max(0.0, min(1.0, w_k)))
        ax.plot(
            xs_line,
            mu_post_k,
            color="black",
            linewidth=2,
            alpha=alpha_k,
            label="post mean (weighted)" if idx == len(order_k) - 1 else None,
        )

    ax.set_xlabel("SPEND")
    ax.set_ylabel("CLICKS")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _run_top_group_for_current_K():
    # Load data
    df = pd.read_csv("tests/data/mmm.csv")
    # Select a single group with enough rows (largest by count)
    gcols = ["ORGANISATION_SUBVERTICAL", "TERRITORY_NAME", "MARKETING_CHANNEL"]
    grp = df.groupby(gcols)
    top_key, top_df = max(grp, key=lambda kv: len(kv[1]))

    xs = jnp.asarray(top_df["SPEND"].to_numpy())
    ys = jnp.asarray(top_df["CLICKS"].to_numpy())

    # Hyperparameters and config
    dir_alpha = jnp.full(K, DIRICHLET_CONC)
    # Priors: keep broad; alpha mean around scale of clicks
    y_mean = float(top_df["CLICKS"].mean())
    shape_a = 2.0
    rate_a = shape_a / max(y_mean, 1.0)
    sigma = float(top_df["CLICKS"].std() * 0.5 + 1e-3)

    key = jax.random.PRNGKey(20251111)
    out = run_mcmc(
        key,
        xs,
        ys,
        dir_alpha=dir_alpha,
        cfg=MCMCConfig(
            num_iters=1500,
            burn_in=500,
            thin=10,
            mh_step_kd=0.03,
            mh_step_n=0.03,
            mh_step_a=0.03,
        ),
        # Use wider priors for flexibility
        shape_kd=4.0,
        rate_kd=0.2,
        shape_n=1.0,
        rate_n=0.5,
        shape_a=shape_a,
        rate_a=rate_a,
        sigma=sigma,
    )

    print(f"[K={K}] group", top_key)
    print("[posterior mean] weights:", out["post_w"])
    print("[posterior mean] Kd:", out["post_kd"])
    print("[posterior mean] n :", out["post_n"])
    print("[posterior mean] alpha:", out["post_a"])
    print("[accept rates] kd, n, alpha:", out["acc_kd"], out["acc_n"], out["acc_a"])

    title = f"3-Param Mix Hill (K={K}): {top_key[0]}_{top_key[1]}_{top_key[2]}"
    out_png = f"mix_hill_mmm_{top_key[0]}_{top_key[1]}_{top_key[2]}_K{K}.png".replace(
        "/", "-"
    )
    _plot_group(
        xs,
        ys,
        out["post_w"],
        out["post_kd"],
        out["post_n"],
        out["post_a"],
        out["last_state"],
        title,
        out_png,
    )


def test_mmm_three_param_mix_hill_once_K2():
    global K
    K = 2
    _run_top_group_for_current_K()


def test_mmm_three_param_mix_hill_once_K3():
    global K
    K = 3
    _run_top_group_for_current_K()


def _run_for_group(key_tuple, group_df):
    xs = jnp.asarray(group_df["SPEND"].to_numpy())
    ys = jnp.asarray(group_df["CLICKS"].to_numpy())

    dir_alpha = jnp.full(K, DIRICHLET_CONC)
    y_mean = float(group_df["CLICKS"].mean())
    shape_a = 2.0
    rate_a = shape_a / max(y_mean, 1.0)
    sigma = float(group_df["CLICKS"].std() * 0.5 + 1e-3)

    key = jax.random.PRNGKey(20251111)
    out = run_mcmc(
        key,
        xs,
        ys,
        dir_alpha=dir_alpha,
        cfg=MCMCConfig(
            num_iters=1500,
            burn_in=500,
            thin=10,
            mh_step_kd=0.03,
            mh_step_n=0.03,
            mh_step_a=0.03,
        ),
        shape_kd=4.0,
        rate_kd=0.2,
        shape_n=1.0,
        rate_n=0.5,
        shape_a=shape_a,
        rate_a=rate_a,
        sigma=sigma,
    )

    print("[group]", key_tuple)
    print("[posterior mean] weights:", out["post_w"])
    print("[posterior mean] Kd:", out["post_kd"])
    print("[posterior mean] n :", out["post_n"])
    print("[posterior mean] alpha:", out["post_a"])
    print("[accept rates] kd, n, alpha:", out["acc_kd"], out["acc_n"], out["acc_a"])

    title = f"3-Param Mix Hill: {key_tuple[0]}_{key_tuple[1]}_{key_tuple[2]}"
    out_png = (
        f"mix_hill_mmm_{key_tuple[0]}_{key_tuple[1]}_{key_tuple[2]}_K{K}.png".replace(
            "/", "-"
        )
    )
    _plot_group(
        xs,
        ys,
        out["post_w"],
        out["post_kd"],
        out["post_n"],
        out["post_a"],
        out["last_state"],
        title,
        out_png,
    )


def test_mmm_three_param_mix_hill_top4():
    """Run the same modeling/plotting for the top 4 groups by sample size.

    Keeps runtime reasonable while covering multiple combinations.
    """
    df = pd.read_csv("tests/data/mmm.csv")
    gcols = ["ORGANISATION_SUBVERTICAL", "TERRITORY_NAME", "MARKETING_CHANNEL"]
    grp = df.groupby(gcols)
    # Pick top 4 groups by size
    groups_sorted = sorted(grp, key=lambda kv: len(kv[1]), reverse=True)[:4]
    for key_tuple, gdf in groups_sorted:
        _run_for_group(key_tuple, gdf)


# (log-log variant removed by request)
