from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
from genjax import (
    Arguments,
    ChoiceMap,
    ChoiceMapBuilder,
    Weight,
    gen,
    normal,
    GenerativeFunction,
)
from jax import jit, random, vmap
from jax.scipy.special import logsumexp
from jax.scipy.stats import gaussian_kde
from jaxtyping import Array, PRNGKeyArray


def get_proposal(vars: set[str]) -> GenerativeFunction:
    @gen
    def proposal(μ: Array, σ: Array):
        for var in vars:
            normal(μ, σ) @ var

    return proposal


@jit
@partial(vmap, in_axes=(0, None, None, None, None, None))
def importance(
    key: PRNGKeyArray,
    constraint: ChoiceMap,
    model: GenerativeFunction,
    model_args: Arguments,
    proposal: GenerativeFunction,
    proposal_args: Arguments,
) -> Weight:
    trace = proposal.simulate(key, args=proposal_args)
    proposal_logpdf = trace.get_score()

    target_logpdf, _ = model.assess(constraint ^ trace.get_sample(), args=model_args)
    return target_logpdf - proposal_logpdf


@gen
def model(μ: Array, σ: Array) -> tuple[Array, Array]:
    x = normal(μ, σ) @ "x"
    y = normal(x**2 / 10, σ / 10) @ "y"
    return x, y


@jit
@partial(vmap, in_axes=(0, 0, 0, None))
def joint(key: PRNGKeyArray, x: Array, y: Array, args: Arguments) -> Weight:
    logpdf, retval = model.assess(ChoiceMap.kw(x=x, y=y), args)
    return logpdf


def get_marginal(variable_name: str, all: set[str]) -> GenerativeFunction:
    @jit
    @partial(vmap, in_axes=(0, 0, None))
    def marginal(
        key: PRNGKeyArray, x: Array, args: Arguments, num: int = 5000
    ) -> Weight:
        keys = random.split(key, num=num)
        weight = importance(
            keys,
            ChoiceMapBuilder[variable_name].set(x),
            model,
            args,
            get_proposal(all - {variable_name}),
            args,
        )
        return logsumexp(weight) - jnp.log(num)

    return marginal


@jit
@partial(vmap, in_axes=(0, None))
def generate_samples(key: PRNGKeyArray, args: Arguments) -> tuple[Array, Array]:
    trace = model.simulate(key, args)
    samples = trace.get_sample()
    return samples["x"], samples["y"]


def test_marginals():
    args = (0.0, 1.0)
    key = random.key(314159)

    # direct sampling
    n_samples = 500
    keys = random.split(key, n_samples)
    key = keys[-1]
    x, y = generate_samples(keys, args)

    keys = random.split(key, n_samples)
    key = keys[-1]
    logp = joint(keys, x, y, args)

    n_grid = 100
    xgrid = jnp.linspace(-3, 3, n_grid)
    ygrid = jnp.linspace(-0.5, 1.0, n_grid)

    px = gaussian_kde(x, bw_method=0.05).evaluate(xgrid)
    py = gaussian_kde(y, bw_method=0.05).evaluate(ygrid)

    plt.figure(figsize=(6, 6))

    plt.tricontourf(x, y, jnp.exp(logp), levels=20, cmap="Greys")
    plt.scatter(x, y, s=1, color="black")

    keys = random.split(key, n_grid)
    key = keys[-1]

    logpy = get_marginal("y", {"x", "y"})(keys, ygrid, args)

    plt.plot(-jnp.exp(logpy), ygrid, color="orange")
    plt.plot(-py, ygrid, color="darkcyan")

    keys = random.split(key, n_grid)
    key = keys[-1]

    logpx = get_marginal("x", {"x", "y"})(keys, xgrid, args)

    plt.plot(xgrid, jnp.exp(logpx), color="orange", label="importance")
    plt.plot(xgrid, px, label="kde", color="darkcyan")

    plt.xlim(-3, 3)
    plt.ylim(-0.5, 1.0)

    plt.legend()
    plt.savefig("plot.png")
    plt.close()
