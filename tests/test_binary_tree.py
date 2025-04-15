from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from genjax import Arguments, ChoiceMap, Const, flip, gen, normal
from genjax._src.core.compiler.interpreters.incremental import Diff
from jax import jit, random, vmap
from jax.scipy.special import logsumexp
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
    y = model(Const(2 * node_id.unwrap() + 1), *args) @ f"branch:right:{node_id.unwrap()}"

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
        return branch.or_else(leaf)(is_branch, model_args, model_args) @ f"branch:{node_id.unwrap()}"


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
    a = normal(0.0, 1.0) @ "leaf:1"
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


def test_gp_extrapolation():
    key = jax.random.PRNGKey(42)

    # Load data
    x_training = jnp.asarray(pd.read_csv("tests/data/x_training.csv").values)
    y_training = jnp.asarray(pd.read_csv("tests/data/y_training.csv").values)

    x_test = jnp.asarray(pd.read_csv("tests/data/x_test.csv").values)
    y_test = jnp.asarray(pd.read_csv("tests/data/y_test.csv").values)

    obs = ChoiceMap.d({"x": x_training, "y": y_training})
    model_args = (
        Const(1),  # initial node_id
        jnp.array(0.3),  # branch_prob
    )

    num_samples = 40000
    key, subkey = jax.random.split(key)
    _, mh_chain = run_inference(model, model_args, obs, subkey, num_samples)

    validate_mh(mh_chain)
