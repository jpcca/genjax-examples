from abc import ABC
from dataclasses import dataclass, field
from functools import partial
from itertools import count

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from genjax import Arguments, ChoiceMap, Const, beta, flip, gen, normal
from genjax._src.core.compiler.interpreters.incremental import Diff
from jax import jit, random, vmap
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray, Scalar

MAX_DEPTH = 5
global depth
depth = 0


@jax.tree_util.register_dataclass
@dataclass
class Interval:
    lower: Float[Array, "..."]
    upper: Float[Array, "..."]


@jax.tree_util.register_dataclass
@dataclass
class Node(ABC):
    pass


@jax.tree_util.register_dataclass
@dataclass
class InternalNode(Node):
    left: Node
    right: Node
    interval: Interval


@jax.tree_util.register_dataclass
@dataclass
class LeafNode(Node):
    value: Float[Array, "..."]
    interval: Interval


def leaf(interval: Interval) -> LeafNode:
    value = normal(0.0, 1.0) @ "value"
    return LeafNode(value, interval)


def branch(interval: Interval) -> InternalNode:

    fraction = beta(2.0, 2.0) @ "fraction"
    midpoint = interval.lower + fraction * (interval.upper - interval.lower)

    left = generate_segments(interval.lower, midpoint) @ "left"
    right = generate_segments(midpoint, interval.upper) @ "right"

    return InternalNode(left, right, interval)


@gen
def generate_segments(lower: Float[Array, "..."], upper: Float[Array, "..."]) -> Node:
    interval = Interval(lower, upper)
    is_leaf = flip(0.7) @ "is_leaf"
    return jax.lax.cond(is_leaf, leaf, branch, interval)


def metropolis_hastings_move(mh_args, key):
    # For now, we give the kernel the full state of the model, the proposal, and the observations.
    trace, model, proposal, proposal_args, observations = mh_args
    model_args = trace.get_args()

    # The core computation is updating a trace, and for that we will call the model's update method.
    # The update method takes a random key, a trace, and a choice map object, and argument difference objects.
    argdiffs = Diff.no_change(model_args)
    proposal_args_forward = (trace, *proposal_args)

    # We sample the proposed changes to the trace.
    # This is encapsulated in a simple GenJAX generative function.
    key, subkey = jax.random.split(key)
    fwd_choices, fwd_weight, _ = proposal.propose(key, proposal_args_forward)

    new_trace, weight, _, discard = model.update(subkey, trace, fwd_choices, argdiffs)

    # Because we are using MH, we don't directly accept the new trace.
    # Instead, we compute a (log) acceptance ratio α and decide whether to accept the new trace, and otherwise keep the old one.
    proposal_args_backward = (new_trace, *proposal_args)
    bwd_weight, _ = proposal.assess(discard, proposal_args_backward)
    α = weight - fwd_weight + bwd_weight
    key, subkey = jax.random.split(key)
    ret_fun = jax.lax.cond(jnp.log(jax.random.uniform(subkey)) < α, lambda: new_trace, lambda: trace)
    return (ret_fun, model, proposal, proposal_args, observations), ret_fun


def mh(trace, model, proposal, proposal_args, observations, key, num_updates):
    mh_keys = jax.random.split(key, num_updates)
    last_carry, mh_chain = jax.lax.scan(
        metropolis_hastings_move,
        (trace, model, proposal, proposal_args, observations),
        mh_keys,
    )
    return last_carry[0], mh_chain


@gen
def prop(tr, *_):
    orig_a = tr.get_choices()["uniform"]
    a = normal(orig_a, 1.0) @ "uniform"
    return a


def custom_mh(trace, model, observations, key, num_updates):
    return mh(trace, model, prop, (), observations, key, num_updates)


def run_inference(model, model_args, obs, key, num_samples):
    key, subkey1, subkey2 = jax.random.split(key, 3)
    # We sample once from a default importance sampler to get an initial trace.
    # The particular initial distribution is not important, as the MH kernel will rejuvenate it.
    tr, _ = model.importance(subkey1, obs, model_args)
    # We then run our custom Metropolis-Hastings kernel to rejuvenate the trace.
    rejuvenated_trace, mh_chain = custom_mh(tr, model, obs, subkey2, num_samples)
    return rejuvenated_trace, mh_chain


def validate_mh(mh_chain):
    a = mh_chain.get_choices()["a"]
    b = mh_chain.get_choices()["b"]
    y = mh_chain.get_retval()
    x = mh_chain.get_args()[0]
    plt.plot(range(len(y)), a * x + b)
    plt.plot(range(len(y)), y, color="k")
    plt.show()


# @jit
@partial(vmap, in_axes=(0, None))
def generate_random_programs(key: PRNGKeyArray, args: Arguments):
    trace = abstract_syntax_tree.simulate(key, args)
    samples = trace.get_sample()
    return samples


def test_gp_extrapolation():
    key = jax.random.PRNGKey(42)

    # Load data
    x_training = jnp.asarray(pd.read_csv("tests/data/x_training.csv").values)
    y_training = jnp.asarray(pd.read_csv("tests/data/y_training.csv").values)

    x_test = jnp.asarray(pd.read_csv("tests/data/x_test.csv").values)
    y_test = jnp.asarray(pd.read_csv("tests/data/y_test.csv").values)

    obs = ChoiceMap.d({"x": x_training, "y": y_training})
    model_args = (
        jnp.amin(x_training),
        jnp.amax(x_training),
    )

    num_samples = 40000
    key, subkey = jax.random.split(key)
    _, mh_chain = run_inference(generate_segments, model_args, obs, subkey, num_samples)

    validate_mh(mh_chain)
