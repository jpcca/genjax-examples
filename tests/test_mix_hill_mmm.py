from typing import NamedTuple, Dict, Tuple

import csv
import collections

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit

from genjax import gen, Const  # type: ignore
from genjax import normal, gamma, dirichlet, mix  # type: ignore

import matplotlib.pyplot as plt


# Number of mixture components
NUM_COMPONENTS = 2


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
# Generative model (K fixed, using mix)
# =====================================================


@gen
def hill_mu_component(xi, kd, n_k):
    mu = hill(xi, kd, n_k)
    return mu


@gen
def mix_hill(
    xs: jnp.ndarray,
    alpha: jnp.ndarray,  # (NUM_COMPONENTS,)
    shape_kd: float = 4.0,
    rate_kd: float = 0.2,
    shape_n: float = 1.0,
    rate_n: float = 0.5,
    sigma: float = 0.05,
):
    # Mixture weights
    weights = dirichlet(alpha) @ "weights"  # (NUM_COMPONENTS,)

    # Cluster parameters (Kd, n)
    Kd = gamma(shape_kd, rate_kd, sample_shape=Const((NUM_COMPONENTS,))) @ (
        "clusters",
        "Kd",
    )  # (NUM_COMPONENTS,)
    n = gamma(shape_n, rate_n, sample_shape=Const((NUM_COMPONENTS,))) @ (
        "clusters",
        "n",
    )  # (NUM_COMPONENTS,)

    # Build mixture with mix combinator and vmap over observations
    logits = jnp.log(weights + 1e-20)  # (NUM_COMPONENTS,)
    hill_mixture = mix(hill_mu_component, hill_mu_component)
    hill_mixture_vm = hill_mixture.vmap(
        in_axes=(
            None,  # logits shared
            (0, None, None),  # args for component 1: (xs, Kd[0], n[0])
            (0, None, None),  # args for component 2: (xs, Kd[1], n[1])
        )
    )

    mu_vec = (
        hill_mixture_vm(
            logits,
            (xs, Kd[0], n[0]),
            (xs, Kd[1], n[1]),
        )
        @ "y_mix"
    )

    # Observations
    y = normal(mu_vec, sigma) @ "y"
    return y


# =====================================================
# Collapsed Gibbs for z & Gibbs for weights
# =====================================================


@jit
def sample_assignments(
    key: jax.Array,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    weights: jnp.ndarray,  # (NUM_COMPONENTS,)
    Kd: jnp.ndarray,  # (NUM_COMPONENTS,)
    n: jnp.ndarray,  # (NUM_COMPONENTS,)
    sigma: float,
) -> Tuple[jax.Array, jnp.ndarray]:
    def logit_row(xi, yi):
        mu_k = hill(jnp.full_like(Kd, xi), Kd, n)  # (NUM_COMPONENTS,)
        ll_k = log_normal_pdf(yi, mu_k, sigma)  # (NUM_COMPONENTS,)
        return _safe_log(weights) + ll_k

    logits = vmap(logit_row)(xs, ys)  # (N, NUM_COMPONENTS)
    key, sub = jax.random.split(key)
    assignments = jax.random.categorical(sub, logits, axis=-1)
    return key, assignments


@jit
def sample_mixture_weights(
    key: jax.Array, alpha: jnp.ndarray, assignments: jnp.ndarray
) -> Tuple[jax.Array, jnp.ndarray]:
    counts = jnp.bincount(assignments, length=NUM_COMPONENTS).astype(alpha.dtype)
    key, sub = jax.random.split(key)
    weights = jax.random.dirichlet(sub, alpha + counts)
    return key, weights


# =====================================================
# MH for (Kd, n) – log-space RW (mask-based, JIT-friendly)
# =====================================================


class HillMHConfig(NamedTuple):
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
    mask = (z == k).astype(ys.dtype)  # (N,)

    def logpost(kd_log_prime):
        kd_prime = jnp.exp(kd_log_prime)
        n_k = jnp.exp(n_log[k])
        log_prior = (shape_kd - 1.0) * kd_log_prime - rate_kd * kd_prime

        mu_all = hill(xs, kd_prime, n_k)  # (N,)
        ll_all = log_normal_pdf(ys, mu_all, sigma)  # (N,)
        ll = jnp.sum(mask * ll_all)
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
    cfg: HillMHConfig,
) -> Tuple[jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Kd_log = jnp.log(Kd)
    n_log = jnp.log(n)

    def upd_k(i, carry):
        key, Kd_log_vec, n_log_vec, acc_kd_vec, acc_n_vec = carry

        key, new_kd_log, acc_kd = _mh_one_param(
            key,
            Kd_log_vec[i],
            cfg.step_kd,
            _logpost_kd_factory(i, xs, ys, z, Kd_log, n_log, sigma, shape_kd, rate_kd),
        )
        Kd_log_vec = Kd_log_vec.at[i].set(new_kd_log)

        key, new_n_log, acc_n = _mh_one_param(
            key,
            n_log_vec[i],
            cfg.step_n,
            _logpost_n_factory(i, xs, ys, z, Kd_log_vec, n_log, sigma, shape_n, rate_n),
        )
        n_log_vec = n_log_vec.at[i].set(new_n_log)

        acc_kd_vec = acc_kd_vec.at[i].add(acc_kd.astype(acc_kd_vec.dtype))
        acc_n_vec = acc_n_vec.at[i].add(acc_n.astype(acc_n_vec.dtype))
        return key, Kd_log_vec, n_log_vec, acc_kd_vec, acc_n_vec

    init = (
        key,
        Kd_log,
        n_log,
        jnp.zeros((NUM_COMPONENTS,)),
        jnp.zeros((NUM_COMPONENTS,)),
    )
    key, Kd_log, n_log, acc_kd, acc_n = lax.fori_loop(0, NUM_COMPONENTS, upd_k, init)
    return key, jnp.exp(Kd_log), jnp.exp(n_log), acc_kd, acc_n


# =====================================================
# Full MCMC driver
# =====================================================


class HillMMMCMCConfig(NamedTuple):
    num_iters: int = 3000
    burn_in: int = 1000
    thin: int = 10
    mh_step_kd: float = 0.05
    mh_step_n: float = 0.05


class HillMMMCMCState(NamedTuple):
    key: jax.Array
    weights: jnp.ndarray  # (NUM_COMPONENTS,)
    z: jnp.ndarray  # (N,)
    Kd: jnp.ndarray  # (NUM_COMPONENTS,)
    n: jnp.ndarray  # (NUM_COMPONENTS,)


@jit
def mcmc_step(
    state: HillMMMCMCState,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    alpha: jnp.ndarray,
    shape_kd: float,
    rate_kd: float,
    shape_n: float,
    rate_n: float,
    sigma: float,
    mh_cfg: HillMHConfig,
) -> Tuple[HillMMMCMCState, Tuple[jnp.ndarray, jnp.ndarray]]:
    key = state.key

    # z | rest (collapsed Gibbs)
    key, z = sample_assignments(key, xs, ys, state.weights, state.Kd, state.n, sigma)

    # w | z (Gibbs)
    key, w = sample_mixture_weights(key, alpha, z)

    # (Kd, n) | rest (MH)
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

    new_state = HillMMMCMCState(key, w, z, Kd_new, n_new)
    return new_state, (acc_kd, acc_n)


def run_hill_mmm_mcmc(
    key: jax.Array,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    alpha: jnp.ndarray,
    shape_kd: float,
    rate_kd: float,
    shape_n: float,
    rate_n: float,
    sigma: float,
    cfg: HillMMMCMCConfig,
) -> Dict[str, jnp.ndarray]:
    # Initialise from prior via GenJAX model
    key, sub = jax.random.split(key)
    tr = mix_hill.simulate(sub, (xs, alpha, shape_kd, rate_kd, shape_n, rate_n, sigma))
    ch = tr.get_choices()
    w0 = ch["weights"]
    Kd0 = ch[("clusters", "Kd")]
    n0 = ch[("clusters", "n")]

    key, z0 = sample_assignments(key, xs, ys, w0, Kd0, n0, sigma)
    state = HillMMMCMCState(key, w0, z0, Kd0, n0)

    mh_cfg = HillMHConfig(cfg.mh_step_kd, cfg.mh_step_n)

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
        st, sum_w, sum_kd, sum_n, acc_kd_sum, acc_n_sum, kept = lax.cond(
            keep_cond,
            do_keep,
            no_keep,
            operand=(st, sum_w, sum_kd, sum_n, acc_kd_sum, acc_n_sum, kept),
        )

        return (st, t, sum_w, sum_kd, sum_n, acc_kd_sum, acc_n_sum, kept), None

    init_carry = (
        state,
        jnp.array(0, dtype=jnp.int32),
        jnp.zeros((NUM_COMPONENTS,)),
        jnp.zeros((NUM_COMPONENTS,)),
        jnp.zeros((NUM_COMPONENTS,)),
        jnp.zeros((NUM_COMPONENTS,)),
        jnp.zeros((NUM_COMPONENTS,)),
        jnp.array(0, dtype=jnp.int32),
    )

    (st, t, sum_w, sum_kd, sum_n, acc_kd_sum, acc_n_sum, kept), _ = lax.scan(
        one_step, init_carry, xs=jnp.arange(cfg.num_iters)
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


run_hill_mmm_mcmc_jit = jit(run_hill_mmm_mcmc, static_argnames=("cfg",))


# =====================================================
# MMM experiment: fit each group in mmm.csv and plot
# =====================================================


def _load_mmm_groups():
    groups = collections.defaultdict(lambda: {"xs": [], "ys": []})
    with open("tests/data/mmm.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["ORGANISATION_SUBVERTICAL"],
                row["TERRITORY_NAME"],
                row["MARKETING_CHANNEL"],
            )
            groups[key]["xs"].append(float(row["SPEND"]))
            groups[key]["ys"].append(float(row["CLICKS"]))
    return groups


def _plot_group_mmm(xs, ys, out, title, out_png, y_scale: float):
    # xs, ys are original (unscaled) data; out is posterior summary on scaled ys
    xs_line = jnp.linspace(float(xs.min()), float(xs.max()), 400)

    # Collect additional samples from last_state to estimate component std
    st = out["last_state"]
    alpha = jnp.ones(NUM_COMPONENTS)
    shape_kd = 4.0
    rate_kd = 0.2
    shape_n = 1.0
    rate_n = 0.5
    sigma = 0.05
    mh_cfg = HillMHConfig(0.03, 0.03)

    n_draws = 200
    mu_samples = jnp.zeros((n_draws, NUM_COMPONENTS, xs_line.shape[0]))

    def body(carry, i):
        st, mu_s = carry
        st, _ = mcmc_step(
            st,
            xs,
            ys / y_scale,
            alpha,
            shape_kd,
            rate_kd,
            shape_n,
            rate_n,
            sigma,
            mh_cfg,
        )
        for k in range(NUM_COMPONENTS):
            mu_k = hill(xs_line, st.Kd[k], st.n[k]) * y_scale
            mu_s = mu_s.at[i, k, :].set(mu_k)
        return (st, mu_s), None

    (st_final, mu_samples), _ = lax.scan(body, (st, mu_samples), xs=jnp.arange(n_draws))

    mu_mean = mu_samples.mean(axis=0)  # (NUM_COMPONENTS, n_x)
    mu_std = mu_samples.std(axis=0)  # (NUM_COMPONENTS, n_x)

    # Sort xs_line for plotting
    order = jnp.argsort(xs_line)
    xs_plot = xs_line[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, s=10, color="orange", edgecolors="none", label="data")

    # Plot each component curve with ±3σ band; weight affects opacity
    post_w = out["post_w"]
    order_k = list(jnp.argsort(post_w))
    for idx, k in enumerate(order_k):
        k = int(k)
        w_k = float(post_w[k])
        alpha_k = float(0.2 + 0.8 * max(0.0, min(1.0, w_k)))
        mean_k = mu_mean[k][order]
        std_k = mu_std[k][order]

        ax.fill_between(
            xs_plot,
            mean_k - 3.0 * std_k,
            mean_k + 3.0 * std_k,
            color="black",
            alpha=0.12 * alpha_k,
            label="component ±3σ" if idx == 0 else None,
        )
        ax.plot(
            xs_plot,
            mean_k,
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


def test_mmm_mixture_hill():
    key = jax.random.PRNGKey(0)
    groups = _load_mmm_groups()

    for (subv, terr, chn), data in groups.items():
        xs = jnp.array(data["xs"])
        ys = jnp.array(data["ys"])

        # Scale clicks to [0,1] range for stable hill modelling, then scale back
        y_scale = float(jnp.max(ys))
        ys_scaled = ys / y_scale

        alpha = jnp.ones(NUM_COMPONENTS)

        # Priors roughly matching the synthetic demo but allowing broader Kd
        shape_kd = 4.0
        rate_kd = 0.2
        shape_n = 1.0
        rate_n = 0.5
        sigma = 0.05

        cfg = HillMMMCMCConfig(
            num_iters=3000,
            burn_in=0,
            thin=5,
            mh_step_kd=0.03,
            mh_step_n=0.03,
        )

        key, sub = jax.random.split(key)
        out = run_hill_mmm_mcmc_jit(
            sub,
            xs,
            ys_scaled,
            alpha=alpha,
            shape_kd=shape_kd,
            rate_kd=rate_kd,
            shape_n=shape_n,
            rate_n=rate_n,
            sigma=sigma,
            cfg=cfg,
        )

        title = f"{subv} | {terr} | {chn}"
        safe_subv = subv.replace(" ", "_")
        safe_terr = terr.replace(" ", "_")
        safe_chn = chn.replace(" ", "_")
        out_png = (
            f"mix_hill_mmm_{safe_subv}_{safe_terr}_{safe_chn}_K{NUM_COMPONENTS}.png"
        )

        _plot_group_mmm(xs, ys, out, title, out_png, y_scale=y_scale)
