import jax
import jax.numpy as jnp
from genjax import gen, normal, ChoiceMap  # type: ignore
from genjax._src.core.compiler.interpreters.incremental import Diff

key = jax.random.key(0)


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
    ret_fun = jax.lax.cond(
        jnp.log(jax.random.uniform(subkey)) < α, lambda: new_trace, lambda: trace
    )
    return (ret_fun, model, proposal, proposal_args, observations), ret_fun


@gen
def prop(tr, *_):
    a = normal(0.0, 1.0)
    return a


def mh(trace, model, proposal, proposal_args, observations, key, num_updates):
    mh_keys = jax.random.split(key, num_updates)
    last_carry, mh_chain = jax.lax.scan(
        metropolis_hastings_move,
        (trace, model, proposal, proposal_args, observations),
        mh_keys,
    )
    return last_carry[0], mh_chain


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


@gen
def model(x):
    a = normal(0.0, 5.0) @ "a"
    b = normal(0.0, 1.0) @ "b"
    y = normal(a * x + b, 1.0) @ "y"
    return y


def test_mcmc():
    key = jax.random.PRNGKey(42)

    obs = ChoiceMap.d({"y": 5.0})
    model_args = (5.0,)

    num_samples = 1
    key, subkey = jax.random.split(key)
    trace, mh_chain = run_inference(model, model_args, obs, subkey, num_samples)
