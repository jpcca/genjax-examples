from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from genjax import Arguments, ChoiceMap, Const, flip, gen, normal
from genjax._src.core.compiler.interpreters.incremental import Diff
from jax import jit, vmap
from jaxtyping import Array, Float, PRNGKeyArray

MAX_DEPTH = 5
global depth
depth = 0


@gen
def leaf(node_id: Const[int], *args: Float[Array, "..."]):
    y = normal(50.0, 1.0) @ f"normal:{node_id.unwrap()}"
    return y


@gen
def branch(node_id: Const[int], *args: Float[Array, "..."]):

    x = model(Const(2 * node_id.unwrap()), *args) @ f"branch:left:{node_id.unwrap()}"
    y = (
        model(Const(2 * node_id.unwrap() + 1), *args)
        @ f"branch:right:{node_id.unwrap()}"
    )

    return x + y


@gen
def model(node_id: Const[int], *args: Float[Array, "..."]):
    model_args = (node_id, *args)
    branch_prob = args[0]

    global depth  # hacky way to avoid infinite recursion
    if depth >= MAX_DEPTH:
        return leaf(*model_args) @ f"leaf:{node_id.unwrap()}"

    else:
        depth += 1
        is_branch = flip(branch_prob) @ f"is_branch:{node_id.unwrap()}"
        return (
            branch.or_else(leaf)(is_branch, model_args, model_args)
            @ f"branch:{node_id.unwrap()}"
        )


@jit
@partial(vmap, in_axes=(0, None))
def simulate(key: PRNGKeyArray, args: Arguments):
    return model.simulate(key, args)


def test_binary_tree():
    keys = jax.random.split(jax.random.PRNGKey(42), 200)
    trace = simulate(
        keys,
        (
            Const(1),  # initial node_id
            jnp.array(0.3),  # branch_prob
        ),
    )

    assert trace.retval.shape == (200,)
