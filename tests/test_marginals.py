from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
from genjax import (
    Arguments,
    ChoiceMap,
    Selection,
    Weight,
    gen,
    normal,
    GenerativeFunction,
)

from genjax._src.core.generative.choice_map import Static
from jax import jit, random, vmap
from jax.scipy.special import logsumexp
from jax.scipy.stats import gaussian_kde
from jaxtyping import Array, PRNGKeyArray

from typing import Callable


def proposal(X: Static) -> GenerativeFunction:
    @gen
    def proposal_function(μ: Array, σ: Array):
        for key in X.mapping.keys():
            normal(μ, σ) @ key

    return proposal_function


def marginal(
    model: GenerativeFunction, x: Selection, num: int = 5000
) -> Callable[[PRNGKeyArray, ChoiceMap, Arguments], Weight]:
    def marginal_function(key: PRNGKeyArray, X: ChoiceMap, args: Arguments) -> Weight:
        keys = random.split(key, num=num)

        weight = importance(
            keys, X.filter(x), model, args, proposal(X.filter(x.complement())), args
        )
        return logsumexp(weight) - jnp.log(num)

    return marginal_function


def pointwise_mutual_information(
    model: GenerativeFunction, x: Selection, y: Selection, num: int = 5000
) -> Callable[[PRNGKeyArray, ChoiceMap, Arguments], Weight]:
    logpx, logpy = marginal(model, x, num=num), marginal(model, y, num=num)
    logpxy = marginal(model, x | y, num=num)

    def pmi_function(key: PRNGKeyArray, X: ChoiceMap, args: Arguments) -> Weight:
        keys_x, keys_y, keys_xy = random.split(key, num=3)

        return (
            logpxy(keys_xy, X, args) - logpx(keys_x, X, args) - logpy(keys_y, X, args)
        )

    return pmi_function


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

    target_logpdf, _ = model.assess(
        constraint.merge(trace.get_choices()), args=model_args
    )
    return target_logpdf - proposal_logpdf


@gen
def model(μ: Array, σ: Array) -> tuple[Array, Array, Array]:
    x = normal(μ, σ) @ "x"
    y = normal(x**2 / 10, σ / 10) @ "y"
    z = normal(μ, σ / 5) @ "z"
    return x, y, z


@jit
@partial(vmap, in_axes=(0, None))
def model_generate_samples(key: PRNGKeyArray, args: Arguments) -> ChoiceMap:
    return model.simulate(key, args).get_choices()


@jit
@partial(vmap, in_axes=(0, None))
def model_joint(X: ChoiceMap, args: Arguments) -> Weight:
    logpdf, retval = model.assess(X, args)
    return logpdf


@jit
@partial(vmap, in_axes=(0, 0, None))
def model_marginal_x(key: PRNGKeyArray, X: ChoiceMap, args: Arguments) -> Weight:
    return marginal(model, Selection.at["x"])(key, X, args)


@jit
@partial(vmap, in_axes=(0, 0, None))
def model_marginal_y(key: PRNGKeyArray, X: ChoiceMap, args: Arguments) -> Weight:
    return marginal(model, Selection.at["y"])(key, X, args)


@jit
@partial(vmap, in_axes=(0, 0, None))
def mutual_information_xy(key: PRNGKeyArray, X: ChoiceMap, args: Arguments) -> Weight:
    return pointwise_mutual_information(model, Selection.at["x"], Selection.at["y"])(
        key, X, args
    )


@jit
@partial(vmap, in_axes=(0, 0, None))
def mutual_information_xz(key: PRNGKeyArray, X: ChoiceMap, args: Arguments) -> Weight:
    return pointwise_mutual_information(model, Selection.at["x"], Selection.at["z"])(
        key, X, args
    )


def test_marginals():
    args = (0.0, 1.0)
    key = random.key(314159)

    # direct sampling
    n_samples = 500
    keys = random.split(key, n_samples)
    key = keys[-1]

    samples = model_generate_samples(keys, args)
    logp = model_joint(samples, args)

    n_grid = 100
    grid = ChoiceMap.kw(
        x=jnp.linspace(-3, 3, n_grid),
        y=jnp.linspace(-0.5, 1.0, n_grid),
        z=jnp.linspace(-3, 3, n_grid),
    )

    px = gaussian_kde(samples["x"], bw_method=0.05).evaluate(grid["x"])
    py = gaussian_kde(samples["y"], bw_method=0.05).evaluate(grid["y"])

    plt.figure(figsize=(6, 6))

    plt.tricontourf(samples["x"], samples["y"], jnp.exp(logp), levels=20, cmap="Greys")
    plt.scatter(samples["x"], samples["y"], s=1, color="black")

    keys = random.split(key, n_grid)
    key = keys[-1]

    logpy = model_marginal_y(keys, grid, args)

    plt.plot(-jnp.exp(logpy), grid["y"], color="orange")
    plt.plot(-py, grid["y"], color="darkcyan")

    keys = random.split(key, n_grid)
    key = keys[-1]

    logpx = model_marginal_x(keys, grid, args)

    plt.plot(grid["x"], jnp.exp(logpx), color="orange", label="importance")
    plt.plot(grid["x"], px, label="kde", color="darkcyan")

    plt.xlim(-3, 3)
    plt.ylim(-0.5, 1.0)

    plt.legend()
    plt.savefig("plot.png")
    plt.close()

    # Test Mutual Information
    keys = random.split(key, n_samples)
    key = keys[-1]

    mi_xy = mutual_information_xy(keys, samples, args).mean()
    print(f"Mutual Information I(X; Y): {mi_xy:.4f}")

    keys = random.split(key, n_samples)
    key = keys[-1]

    mi_xz = mutual_information_xz(keys, samples, args).mean()
    print(f"Mutual Information I(X; Z): {mi_xz:.4f}")

    # I(X;Z) should be close to 0 as they are independent in the model
    assert jnp.isclose(mi_xz, 0.0, atol=1e-2)
    # I(X;Y) should be positive as Y depends on X
    assert mi_xy > 0.0
